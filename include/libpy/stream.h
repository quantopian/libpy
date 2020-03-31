#pragma once

#include <array>
#include <cstring>
#include <iostream>
#include <string_view>

#include "libpy/call_function.h"
#include "libpy/detail/python.h"
#include "libpy/owned_ref.h"

namespace py {
/** A stream buffer that writes to a Python file-like object.
 */
template<typename CharT, typename Traits = std::char_traits<CharT>>
class basic_streambuf : public std::basic_streambuf<CharT, Traits> {
private:
    using base = std::basic_streambuf<CharT, Traits>;
    using int_type = typename base::int_type;

    /** The reference to the file-like object.
     */
    py::owned_ref<> m_file_ob;

    std::array<CharT, 8192> m_write_buffer;
    std::size_t m_write_ix = 0;

protected:
    /** Write a character to the file-like object.
     */
    virtual int_type overflow(int_type ch) override {
        if (ch == Traits::eof()) {
            return ch;
        }

        m_write_buffer[m_write_ix++] = ch;
        if (m_write_ix == m_write_buffer.size() && sync()) {
            throw exception();
        }

        return ch;
    }

    /** Flush the data to the Python file.

        Note: This does not actually flush the underlying Python file-like object, it just
        commits the C++ buffered writes to the Python object.
     */
    virtual int sync() override {
        if (m_write_ix) {
            auto view = std::string_view(m_write_buffer.data(), m_write_ix);
            auto result = call_method(m_file_ob.get(), "write", view);
            if (!result) {
                return -1;
            }
        }

        m_write_ix = 0;
        return 0;
    }

public:
    explicit basic_streambuf(const py::borrowed_ref<>& file)
        : m_file_ob(py::owned_ref<>::new_reference(file)) {}
};

/** A C++ output stream which writes to a Python file-like object.
 */
template<typename CharT, typename Traits = std::char_traits<CharT>>
class basic_ostream : public std::basic_ostream<CharT, Traits> {
private:
    using ostream = std::basic_ostream<CharT, Traits>;

    basic_streambuf<CharT, Traits> m_buf;

public:
    basic_ostream(py::borrowed_ref<> file) : std::ios(0), m_buf(file) {
        this->rdbuf(&m_buf);
    }

    virtual ~basic_ostream() {
        // behave like a file and flush ourselves on destruction
        m_buf.pubsync();
    }
};

/** A C++ output stream which writes to a Python file-like object.
 */
using ostream = basic_ostream<char>;
}  // namespace py
