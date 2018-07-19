#include <memory>
#include <unordered_map>

#include "zip.h"

#include "libpy/scoped_ref.h"

namespace py::zipfile {
class archive {
private:
    zip_t* m_zip;

    struct close_file {
        void operator()(zip_file_t* f) {
            if (f) {
                zip_fclose(f);
            }
        }
    };

public:
    using file = std::unique_ptr<zip_file_t, close_file>;
    using stat = zip_stat_t;

    archive(const char* path, int flags = ZIP_RDONLY) : m_zip(nullptr) {
        int err = 0;
        m_zip = zip_open(path, flags, &err);
        if (err) {
            zip_error_t error_ob;
            zip_error_init_with_code(&error_ob, err);
            std::runtime_error exc(std::string(zip_error_strerror(&error_ob)));
            zip_error_fini(&error_ob);
            throw exc;
        }
    }

    std::size_t size() {
        auto size = zip_get_num_entries(m_zip, /* flags */ 0);
        if (size < 0) {
            throw std::runtime_error("failed to get size of zipfile");
        }

        return static_cast<std::size_t>(size);
    }

    file open_file(std::size_t ix) {
        return file(zip_fopen_index(m_zip, ix, ZIP_FL_UNCHANGED));
    }

    stat stat_file(std::size_t ix) {
        stat st;
        if (zip_stat_index(m_zip, ix, ZIP_FL_UNCHANGED, &st) < 0) {
            throw std::runtime_error("failed to stat the file");
        }
        return st;
    }

    std::pair<std::string, py::scoped_ref<PyObject>> read_as_pybytes(std::size_t ix) {
        stat st = stat_file(ix);
        auto out = py::scoped_ref(PyBytes_FromStringAndSize(nullptr, st.size));
        if (out) {
            if (zip_fread(open_file(ix).get(), PyBytes_AS_STRING(out.get()), st.size) <
                0) {
                return {st.name, nullptr};
            }
        }
        return {st.name, std::move(out)};
    }

    std::pair<std::string, std::string> read_as_string(std::size_t ix) {
        stat st = stat_file(ix);
        std::string out(st.size, '\0');
        if (zip_fread(open_file(ix).get(), out.data(), st.size) < 0) {
            return {st.name, nullptr};
        }
        return {st.name, std::move(out)};
    }

    ~archive() {
        if (m_zip) {
            zip_discard(m_zip);
        }
    }
};

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

std::unordered_map<std::string, std::string>
read(const std::string& path) {
    archive z(path.data());
    std::size_t entries = z.size();

    std::unordered_map<std::string, std::string> out;

    for (std::size_t n = 0; n < entries; ++n) {
        out.emplace(z.read_as_string(n));
    }

    return out;
}
}  // namespace py::zipfile
