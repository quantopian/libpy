#include <memory>
#include <unordered_map>

#include "zip.h"

#include "libpy/scoped_ref.h"

namespace py::zipfile {
/** A zip archive represents a collection of zip entries.
 */
class archive {
private:
    zip_t* m_zip;

    /** Functor for the entry `unique_ptr`.
     */
    struct close_entry {
        void operator()(zip_file_t* f) {
            if (f) {
                zip_fclose(f);
            }
        }
    };

public:
    using entry = std::unique_ptr<zip_file_t, close_entry>;
    using stat_result = zip_stat_t;

private:
    void maybe_throw_error_code(int err) const {
        if (!err) {
            return;
        }
        zip_error_t error_ob;
        zip_error_init_with_code(&error_ob, err);
        throw zip_error_to_exception(&error_ob);
    }

    std::runtime_error zip_error_to_exception(zip_error_t* error) const {
        std::runtime_error exc(std::string(zip_error_strerror(error)));
        zip_error_fini(error);
        return exc;
    }

public:
    /** An iterator over an archive produces pairs of entry name and the (potentially
     * decompressed) contents of the entry.
     */
    class iterator {
    private:
        std::size_t m_ix;
        archive& m_archive;

    protected:
        friend class archive;

        iterator(std::size_t ix, archive& archive) : m_ix(ix), m_archive(archive) {}

    public:
        iterator& operator++() {
            ++m_ix;
            return *this;
        }

        bool operator==(const iterator& other) const {
            return m_archive.get() == other.m_archive.get() && m_ix == other.m_ix;
        }

        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }

        std::pair<std::string, std::string> operator*() {
            return m_archive.read_as_string(m_ix);
        }
    };

    archive(const std::string& path, int flags = ZIP_RDONLY) : m_zip(nullptr) {
        int err = 0;
        m_zip = zip_open(path.data(), flags, &err);
        maybe_throw_error_code(err);
    }

    zip_t* get() {
        return m_zip;
    }

    const zip_t* get() const {
        return m_zip;
    }

    /** The number of entries in this archive.
     */
    std::size_t size() {
        auto size = zip_get_num_entries(m_zip, /* flags */ 0);
        if (size < 0) {
            throw zip_error_to_exception(zip_get_error(m_zip));
        }

        return static_cast<std::size_t>(size);
    }

    iterator begin() {
        return iterator(0, *this);
    }

    iterator end() {
        return iterator(size(), *this);
    }

    /** Open an entry for reading.
     */
    entry open(std::size_t ix) {
        return entry(zip_fopen_index(m_zip, ix, ZIP_FL_UNCHANGED));
    }

    /** Stat an entry.
     */
    stat_result stat(std::size_t ix) {
        stat_result st;
        if (zip_stat_index(m_zip, ix, ZIP_FL_UNCHANGED, &st) < 0) {
            throw zip_error_to_exception(zip_get_error(m_zip));
        }
        return st;
    }

    /** Read an entry's contents as a Python bytes object.

        @param ix The index of the entry to read.
        @return A C++ tuple of the name of the entry and a new scoped reference to the
                results.
     */
    std::pair<std::string, py::scoped_ref<PyObject>> read_as_pybytes(std::size_t ix) {
        stat_result st = stat(ix);
        auto out = py::scoped_ref(PyBytes_FromStringAndSize(nullptr, st.size));
        if (!out) {
            throw py::exception();
        }

        auto entry = open(ix);
        if (zip_fread(entry.get(), PyBytes_AS_STRING(out.get()), st.size) < 0) {
            throw zip_error_to_exception(zip_file_get_error(entry.get()));
        }
        return {st.name, std::move(out)};
    }

    std::pair<std::string, std::string> read_as_string(std::size_t ix) {
        stat_result st = stat(ix);
        std::string out(st.size, '\0');
        auto entry = open(ix);
        if (zip_fread(entry.get(), out.data(), st.size) < 0) {
            throw zip_error_to_exception(zip_file_get_error(entry.get()));
        }
        return {st.name, std::move(out)};
    }

    ~archive() {
        if (m_zip) {
            zip_discard(m_zip);
        }
    }
};

/** Eagerly read all of the contents from the zipfile at `path` as Python bytes objects.

    This function is meant to be exported to Python with `py::automethod`.
 */
std::unordered_map<std::string, py::scoped_ref<PyObject>>
pymethod_read(PyObject*, const std::string& path) {
    archive z(path.data());
    std::size_t entries = z.size();

    std::unordered_map<std::string, py::scoped_ref<PyObject>> out;

    for (std::size_t n = 0; n < entries; ++n) {
        out.emplace(z.read_as_pybytes(n));
    }

    return out;
}

/** Eagerly read all of the contents from the zipfile at `path`.

    To lower memory pressure, it is often better to iterate over the `archive` object
    directly; however, this can be useful for parallizing across the entries of an
    `archive` because the `archive` access is not thread safe.

    @param `path` The path to read.
    @return The (potentially decompressed) contents of each entry keyed by their name.
 */
std::unordered_map<std::string, std::string>
read(const std::string& path) {
    archive z(path);
    std::size_t entries = z.size();

    std::unordered_map<std::string, std::string> out;

    for (std::size_t n = 0; n < entries; ++n) {
        out.emplace(z.read_as_string(n));
    }

    return out;
}
}  // namespace py::zipfile
