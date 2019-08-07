#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

#include "libpy/char_sequence.h"
#include "libpy/detail/csv_writer.h"
#include "libpy/itertools.h"
#include "libpy/stream.h"
#include "libpy/util.h"

namespace py::csv::writer {
namespace {
/** A rope-backed buffer for building up CSVs in memory.
 */
class rope_adapter {
private:
    std::vector<std::string> m_buffers;
    std::size_t m_size = 0;

public:
    /** Move a string into the rope.

        @param data The string to move into the rope.
        @param size The size of the semantic data in this string.
     */
    void write(std::string&& data, std::size_t size) {
        data.erase(data.begin() + size, data.end());
        m_buffers.emplace_back(std::move(data));
        m_size += size;
    }

    /** Write a string into the rope. The state of data in `data` after `write` is
        unspecified, but the size is unchanged.

        @param data The string to write.
        @param size The size of the semantic data in this string.
     */
    void write(std::string& data, std::size_t size) {
        std::string entry(data.size(), '\0');
        std::swap(data, entry);
        write(std::move(entry), size);
    }

    /** Read the contents of this rope into a flat buffer.

        @param begin The start of the buffer to write into.
        @param end The end of the buffer to write into.
     */
    void read(char* begin, char* end) {
        std::ptrdiff_t size = end - begin;
        std::size_t buffer_ix = 0;
        while (begin < end) {
            const std::string& buffer = m_buffers[buffer_ix++];
            std::size_t to_read_from_buffer = std::min<std::size_t>(size, buffer.size());
            std::memcpy(begin, buffer.data(), to_read_from_buffer);
            size -= to_read_from_buffer;
            begin += to_read_from_buffer;
        }
    }

    std::size_t size() const {
        return m_size;
    }
};

/** Wrapper to present a `write(std::string&, std::size_t)` interface to a `std::ostream`.
 */
template<typename T>
class ostream_adapter {
private:
    T& m_stream;

public:
    ostream_adapter(T& stream) : m_stream(stream) {}

    void write(const std::string& data, std::size_t size) {
        m_stream.write(data.data(), size);
    }
};

/** A buffer over an output stream, which may either be an `ostream_adapter` or
    `rope_adapter`. This object manages formatting objects into large blocks of memory
    before delegating to the underlying stream to write these larger buffers.
 */
template<typename T>
class iobuffer {
private:
    T& m_stream;
    std::string m_buffer;
    std::size_t m_ix;
    int m_float_sigfigs;

    void consume(std::size_t amount) {
        m_ix += amount;
    }

    std::size_t space_left() const {
        return m_buffer.size() - m_ix;
    }

public:
    iobuffer(T& stream, std::size_t buffer_size, std::uint8_t float_sigfigs)
        : m_stream(stream),
          m_buffer(buffer_size, '\0'),
          m_ix(0),
          m_float_sigfigs(float_sigfigs) {
        if (buffer_size % 4096) {
            throw util::formatted_error<std::invalid_argument>(
                "buffer_size must be a multiple of 4096, got:", buffer_size);
        }

        if (!buffer_size) {
            throw std::invalid_argument("buffer_size cannot be 0");
        }

        if (float_sigfigs == 0 || float_sigfigs > 17) {
            throw util::formatted_error<std::invalid_argument>(
                "float_sigfigs must be in the range [1, 17], got: ", float_sigfigs);
        }
    }

    void flush() {
        m_stream.write(m_buffer, m_ix);
        m_ix = 0;
    }

    void write_quoted(const std::string_view& view) {
        write('"');
        for (char c : view) {
            if (c == '"') {
                write('\\');
            }
            else if (c == '\\') {
                write('\\');
            }
            write(c);
        }
        write('"');
    }

    void write(std::string&& data) {
        if (space_left() < data.size()) {
            flush();
            if (m_buffer.size() < data.size()) {
                m_stream.write(std::move(data), data.size());
                return;
            }
        }
        std::memcpy(m_buffer.data() + m_ix, data.data(), data.size());
        m_ix += data.size();
    }

    void write(char c) {
        if (!space_left()) {
            flush();
        }
        m_buffer[m_ix++] = c;
    }

    void write(bool b) {
        write('0' + b);
    }

    void write(double f) {
        auto write = [&] {
            return std::snprintf(m_buffer.data() + m_ix,
                                 m_buffer.size() - m_ix,
                                 "%.*g",
                                 m_float_sigfigs,
                                 f);
        };
        int written = write();
        if (written >= static_cast<std::int64_t>(m_buffer.size() - m_ix)) {
            flush();
            write();
        }
        m_ix += written;
    }

    void write(std::int64_t v) {
        using namespace py::cs::literals;
        constexpr std::int64_t max_int_size = "-9223372036854775808"_arr.size();
        if (space_left() < max_int_size) {
            flush();
        }
        auto begin = m_buffer.data() + m_ix;
        auto [p, errc] = std::to_chars(begin, m_buffer.data() + m_buffer.size(), v);
        m_ix += p - begin;
    }

    template<typename unit>
    void write(const py::datetime64<unit>& dt) {
        if (space_left() < py::detail::max_size<unit>) {
            flush();
        }
        auto begin = m_buffer.data() + m_ix;
        auto [p, errc] = py::to_chars(begin, m_buffer.data() + m_buffer.size(), dt, true);
        consume(p - begin);
    }

    ~iobuffer() {
        flush();
    }
};

template<typename T>
void format_any(iobuffer<T>& buf, const py::any_ref& value) {
    std::stringstream stream;
    stream << value;
    buf.write(stream.str());
}

template<typename T>
void format_pyobject(iobuffer<T>& buf, const py::any_ref& any_value) {
    const auto& as_ob = *reinterpret_cast<const py::scoped_ref<>*>(any_value.addr());
    if (as_ob.get() == Py_None) {
        return;
    }
    // We're giving pandas None values in object dtype columns, and it's giving
    // us back NaN. Instead of updating all possible call sites that might
    // produce that, we allow NaN here as a possible input, handling as it we
    // do for None.
    if (Py_TYPE(as_ob.get()) == &PyFloat_Type &&
        PyFloat_AsDouble(as_ob.get()) != PyFloat_AsDouble(as_ob.get())) {
        return;
    }

    std::string_view text = py::util::pystring_to_string_view(as_ob);
    buf.write_quoted(text);
}

template<typename T, typename F>
void format_float(iobuffer<T>& buf, const py::any_ref& any_value) {
    const auto& as_float = *reinterpret_cast<const F*>(any_value.addr());
    if (as_float != as_float) {
        return;
    }
    buf.write(as_float);
}

template<typename T, typename I>
void format_int(iobuffer<T>& buf, const py::any_ref& any_value) {
    std::int64_t as_int = *reinterpret_cast<const I*>(any_value.addr());
    buf.write(as_int);
}

template<typename T, typename unit>
void format_datetime64(iobuffer<T>& buf, const py::any_ref& any_value) {
    const auto& as_M8 = *reinterpret_cast<const py::datetime64<unit>*>(any_value.addr());
    if (as_M8.isnat()) {
        return;
    }
    buf.write(as_M8);
}

template<typename T>
void format_pybool(iobuffer<T>& buf, const py::any_ref& any_value) {
    const auto& as_pybool = *reinterpret_cast<py::py_bool*>(any_value.addr());
    buf.write(as_pybool);
}

template<typename T>
using format_function = void (*)(iobuffer<T>&, const py::any_ref&);

template<typename T>
std::vector<format_function<T>>
get_format_functions(std::vector<py::array_view<py::any_ref>>& columns) {
    std::size_t num_rows = columns[0].size();
    std::vector<format_function<T>> formatters;
    for (const auto& column : columns) {
        if (column.size() != num_rows) {
            throw std::runtime_error("mismatched column lengths");
        }

        const auto& vtable = column.vtable();
        if (vtable == py::any_vtable::make<double>()) {
            formatters.emplace_back(format_float<T, double>);
        }
        else if (vtable == py::any_vtable::make<float>()) {
            formatters.emplace_back(format_float<T, float>);
        }
        else if (vtable == py::any_vtable::make<std::int64_t>()) {
            formatters.emplace_back(format_int<T, std::int64_t>);
        }
        else if (vtable == py::any_vtable::make<std::int32_t>()) {
            formatters.emplace_back(format_int<T, std::int32_t>);
        }
        else if (vtable == py::any_vtable::make<std::int16_t>()) {
            formatters.emplace_back(format_int<T, std::int16_t>);
        }
        else if (vtable == py::any_vtable::make<std::int8_t>()) {
            formatters.emplace_back(format_int<T, std::int8_t>);
        }
        else if (vtable == py::any_vtable::make<std::uint64_t>()) {
            formatters.emplace_back(format_int<T, std::uint64_t>);
        }
        else if (vtable == py::any_vtable::make<std::uint32_t>()) {
            formatters.emplace_back(format_int<T, std::uint32_t>);
        }
        else if (vtable == py::any_vtable::make<std::uint16_t>()) {
            formatters.emplace_back(format_int<T, std::uint16_t>);
        }
        else if (vtable == py::any_vtable::make<std::uint8_t>()) {
            formatters.emplace_back(format_int<T, std::uint8_t>);
        }
        else if (vtable == py::any_vtable::make<py::scoped_ref<>>()) {
#if PY_MAJOR_VERSION != 2
            // `py::util::pystring_to_string_view` may allocate memory to hold the utf8
            // form. This is cached, so future calls to
            // `py::util::pystring_to_string_view` will return views over the same memory,
            // thus making those future calls safe without holding the GIL
            for (const py::scoped_ref<>& maybe_string : column.cast<py::scoped_ref<>>()) {
                if (maybe_string.get() != Py_None) {
                    py::util::pystring_to_string_view(maybe_string);
                }
            }
#endif
            formatters.emplace_back(format_pyobject<T>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::ns>>()) {
            formatters.emplace_back(format_datetime64<T, py::chrono::ns>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::us>>()) {
            formatters.emplace_back(format_datetime64<T, py::chrono::us>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::ms>>()) {
            formatters.emplace_back(format_datetime64<T, py::chrono::ms>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::s>>()) {
            formatters.emplace_back(format_datetime64<T, py::chrono::s>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::m>>()) {
            formatters.emplace_back(format_datetime64<T, py::chrono::m>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::h>>()) {
            formatters.emplace_back(format_datetime64<T, py::chrono::h>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::D>>()) {
            formatters.emplace_back(format_datetime64<T, py::chrono::D>);
        }
        else {
            formatters.emplace_back(format_any<T>);
        }
    }

    return formatters;
}

template<typename T>
void write_header(iobuffer<T>& buf, const std::vector<std::string>& column_names) {
    auto names_it = column_names.begin();
    buf.write_quoted(*names_it);
    for (++names_it; names_it != column_names.end(); ++names_it) {
        buf.write(',');
        buf.write_quoted(*names_it);
    }
    buf.write('\n');
}

template<typename T>
void write_worker_impl(iobuffer<T>& buf,
                       std::vector<py::array_view<py::any_ref>>& columns,
                       std::int64_t begin,
                       std::int64_t end,
                       const std::vector<format_function<T>>& formatters) {
    for (std::int64_t ix = begin; ix < end; ++ix) {
        auto columns_it = columns.begin();
        auto format_it = formatters.begin();
        (*format_it)(buf, (*columns_it)[ix]);
        for (++columns_it, ++format_it; columns_it != columns.end();
             ++columns_it, ++format_it) {
            buf.write(',');
            (*format_it)(buf, (*columns_it)[ix]);
        }
        buf.write('\n');
    }
}

template<typename T>
void write_worker(std::mutex* exception_mutex,
                  std::vector<std::exception_ptr>* exceptions,
                  iobuffer<T>* buf,
                  std::vector<py::array_view<py::any_ref>>* columns,
                  std::int64_t begin,
                  std::int64_t end,
                  const std::vector<format_function<T>>* formatters) {
    try {
        write_worker_impl<T>(*buf, *columns, begin, end, *formatters);
    }
    catch (const std::exception&) {
        std::lock_guard<std::mutex> guard(*exception_mutex);
        exceptions->emplace_back(std::current_exception());
    }
}

PyObject* write_in_memory(const std::vector<std::string>& column_names,
                          std::vector<py::array_view<py::any_ref>>& columns,
                          std::size_t buffer_size,
                          int num_threads,
                          std::uint8_t float_sigfigs) {
    if (columns.size() != column_names.size()) {
        throw std::runtime_error("mismatched column_names and columns");
    }

    if (!columns.size()) {
        return py::to_object("").escape();
    }

    if (num_threads <= 0) {
        num_threads = 1;
    }

    std::size_t num_rows = columns[0].size();
    auto formatters = get_format_functions<rope_adapter>(columns);

    std::vector<rope_adapter> streams(num_threads);
    {
        std::vector<iobuffer<rope_adapter>> bufs;
        for (auto& stream : streams) {
            bufs.emplace_back(stream, buffer_size, float_sigfigs);
        }

        write_header(bufs[0], column_names);

        if (num_threads <= 1) {
            write_worker_impl(bufs[0], columns, 0, num_rows, formatters);
        }
        else {
            std::mutex exception_mutex;
            std::vector<std::exception_ptr> exceptions;

            std::size_t group_size = num_rows / num_threads + 1;
            std::vector<std::thread> threads;
            for (int n = 0; n < num_threads; ++n) {
                std::int64_t begin = n * group_size;
                threads.emplace_back(
                    std::thread(write_worker<rope_adapter>,
                                &exception_mutex,
                                &exceptions,
                                &bufs[n],
                                &columns,
                                begin,
                                std::min<std::int64_t>(begin + group_size, num_rows),
                                &formatters));
            }

            for (auto& thread : threads) {
                thread.join();
            }

            for (auto& e : exceptions) {
                std::rethrow_exception(e);
            }
        }
    }

    std::vector<std::size_t> sizes;
    std::size_t outsize = 0;
    for (auto& stream : streams) {
        sizes.emplace_back(stream.size());
        outsize += sizes.back();
    }

    char* underlying_buffer;
#if PY_MAJOR_VERSION == 2
    py::scoped_ref out(PyString_FromStringAndSize(nullptr, outsize));
    if (!out) {
        return nullptr;
    }
    underlying_buffer = PyString_AS_STRING(out.get());
#else
    py::scoped_ref out(PyBytes_FromStringAndSize(nullptr, outsize));
    if (!out) {
        return nullptr;
    }
    underlying_buffer = PyBytes_AS_STRING(out.get());
#endif

    std::size_t ix = 0;
    for (auto [stream, size] : py::zip(streams, sizes)) {
        stream.read(underlying_buffer + ix, underlying_buffer + ix + size);
        ix += size;
    }

    return std::move(out).escape();
}
}  // namespace

/** Format a CSV from an array of columns.

    @param stream The ostream to write into.
    @param column_names The names to write into the column header.
    @param columns The arrays of values for each column. Columns are written in the order
                   they appear here, and  must be aligned with `column_names`.
    @param buffer_size The number of bytes to buffer between calls to `stream.write`.
                       This must be a power of 2 greater than or equal to 2 ** 8.
    @param float_sigfigs The number of significant figures to print floats with.
*/
void write(std::ostream& stream,
           const std::vector<std::string>& column_names,
           std::vector<py::array_view<py::any_ref>>& columns,
           std::size_t buffer_size,
           std::uint8_t float_sigfigs) {
    if (columns.size() != column_names.size()) {
        throw std::runtime_error("mismatched column_names and columns");
    }

    if (!columns.size()) {
        return;
    }

    std::size_t num_rows = columns[0].size();
    auto formatters = get_format_functions<ostream_adapter<std::ostream>>(columns);
    ostream_adapter stream_adapter(stream);
    iobuffer<ostream_adapter<std::ostream>> buf(stream_adapter,
                                                buffer_size,
                                                float_sigfigs);
    write_header(buf, column_names);
    write_worker_impl(buf, columns, 0, num_rows, formatters);
}

/** Format a CSV from an array of columns. This is meant to be exposed to Python with
    `py::automethod`.

    @param file A python object which is either a string to be interpreted as a file name,
                or None, in which case the data will be returned as a Python string.
    @param column_names The names to write into the column header.
    @param columns The arrays of values for each column. Columns are written in the order
                   they appear here, and  must be aligned with `column_names`.
    @param float_sigfigs The number of significant figures to print floats with.
    @return Either the data as a Python string, or None.
*/
PyObject* py_write(PyObject*,
                   const py::scoped_ref<>& file,
                   const std::vector<std::string>& column_names,
                   std::vector<py::array_view<py::any_ref>>& columns,
                   std::size_t buffer_size,
                   int num_threads,
                   std::uint8_t float_sigfigs) {
    if (file.get() == Py_None) {
        return write_in_memory(column_names,
                               columns,
                               buffer_size,
                               num_threads,
                               float_sigfigs);
    }
    else if (num_threads > 1) {
        py::raise(PyExc_ValueError)
            << "cannot pass num_threads > 1 with file-backed output";
        return nullptr;
    }
    else if (PyUnicode_Check(file)) {
        const char* text = py::util::pystring_to_cstring(file);
        if (!text) {
            return nullptr;
        }
        std::ofstream stream(text, std::ios::binary);
        if (!stream) {
            py::raise(PyExc_OSError) << "failed to open file";
            return nullptr;
        }
        write(stream, column_names, columns, buffer_size, float_sigfigs);
        if (!stream) {
            py::raise(PyExc_OSError) << "failed to write csv";
            return nullptr;
        }
        Py_RETURN_NONE;
    }
    else {
        py::ostream stream(file);
        write(stream, column_names, columns, buffer_size, float_sigfigs);
        if (!stream) {
            py::raise(PyExc_OSError) << "failed to write csv";
            return nullptr;
        }
        Py_RETURN_NONE;
    }
}
}  // namespace py::csv::writer
