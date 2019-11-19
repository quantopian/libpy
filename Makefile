# Put custom environment stuff here.
-include Makefile.local

PYTHON ?= python
PYTEST ?= pytest

MAJOR_VERSION := 1
MINOR_VERSION := 0
MICRO_VERSION := 0

PY_VERSION := $(shell $(PYTHON) etc/python_version.py)
PY_MAJOR_VERSION := $(word 1,$(PY_VERSION))
PY_MINOR_VERSION := $(word 2,$(PY_VERSION))

CLANG_TIDY ?= clang-tidy
CLANG_FORMAT ?= clang-format
GTEST_BREAK ?= 1

OPTLEVEL ?= 3
MAX_ERRORS ?= 5
WARNINGS := -Werror -Wall -Wextra -Wno-register -Wno-missing-field-initializers \
	-Wsign-compare -Wsuggest-override -Wparentheses -Waggressive-loop-optimizations \
	-Wno-class-memaccess
CXXFLAGS = -std=gnu++17 -g -O$(OPTLEVEL) \
	-fwrapv -fno-strict-aliasing -pipe \
	-march=x86-64 -mtune=generic \
	-fvisibility=hidden
	-fmax-errors=$(MAX_ERRORS) $(WARNINGS) \
	-DPY_MAJOR_VERSION=$(PY_MAJOR_VERSION) \
	-DPY_MINOR_VERSION=$(PY_MINOR_VERSION)

# https://github.com/quantopian/libpy/pull/86/files#r309288697
INCLUDE_DIRS := include/ \
	$(shell $(PYTHON) -c "from distutils import sysconfig; \
						  print(sysconfig.get_config_var('INCLUDEPY'))") \
	$(shell $(PYTHON) -c 'import numpy as np;print(np.get_include())')
INCLUDE := $(foreach d,$(INCLUDE_DIRS), -I$d)

SO_SUFFIX := $(shell $(PYTHON) etc/ext_suffix.py)
LIBRARY := py
SHORT_SONAME := lib$(LIBRARY)$(SO_SUFFIX)
SONAME := $(SHORT_SONAME).$(MAJOR_VERSION).$(MINOR_VERSION).$(MICRO_VERSION)
OS := $(shell uname)
ifeq ($(OS),Darwin)
	SONAME_FLAG := install_name
	SONAME_PATH := @rpath/$(SONAME)
	AR := libtool
	ARFLAGS := -static -o
	LDFLAGS += -undefined dynamic_lookup
	LD_PRELOAD_VAR := DYLD_INSERT_LIBRARIES
else
	CXXFLAGS += -fstack-protector-strong
	SONAME_FLAG := soname
	SONAME_PATH := $(SONAME)
	LDFLAGS += $(shell $(PYTHON)-config --ldflags)
	LD_PRELOAD_VAR := LD_PRELOAD
endif

# Sanitizers
ASAN_OPTIONS := symbolize=1
LSAN_OPTIONS := suppressions=testleaks.supp
ASAN_SYMBOLIZER_PATH ?= llvm-symbolizer

SANITIZE_ADDRESS ?= 0
ifneq ($(SANITIZE_ADDRESS),0)
	OPTLEVEL := 0
	CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address
	TEST_LD_PRELOAD += $(shell CXX=$(CXX) etc/asan-path)
endif

ifneq ($(OPTLEVEL),0)
	CXXFLAGS += -flto
	LDFLAGS += -flto
endif

GCC_TIME_REPORT ?= 0
ifneq ($(GCC_TIME_REPORT),0)
	CXXFLAGS += -ftime-report -ftime-report-details
endif

GCC_TRACE ?= 0
ifneq ($(GCC_TRACE),0)
	CXXFLAGS += -Q
endif


SANITIZE_UNDEFINED ?= 0
ifneq ($(SANITIZE_UNDEFINED),0)
	OPTLEVEL := 0
	CXXFLAGS += -fsanitize=undefined
	LDFLAGS += -lubsan
endif

SOURCES := $(wildcard src/*.cc)
OBJECTS := $(SOURCES:.cc=.o)
DFILES :=  $(SOURCES:.cc=.d)

GTEST_OUTPUT ?=
GTEST_ROOT:= submodules/googletest
GTEST_DIR := $(GTEST_ROOT)/googletest
GTEST_HEADERS := $(wildcard $(GTEST_DIR)/include/gtest/*.h) \
	$(wildcard $(GTEST_DIR)/include/gtest/internal/*.h)
GTEST_SRCS := $(wildcard $(GTEST_DIR)/src/*.cc) \
	$(wildcard $(GTEST_DIR)/src/*.h) $(GTEST_HEADERS)
GTEST_FILTER ?= '*'

TEST_SOURCES := $(wildcard tests/*.cc)
TEST_DFILES := $(TEST_SOURCES:.cc=.d)
TEST_OBJECTS := $(TEST_SOURCES:.cc=.o)
TEST_HEADERS := $(wildcard tests/*.h) $(GTEST_HEADERS)
TEST_INCLUDE := -I tests -I $(GTEST_DIR)/include
TEST_MODULE := tests/_runner$(SO_SUFFIX)
PYTHON_TESTS := $(wildcard tests/*.py)

ALL_SOURCES := $(SOURCES) $(TEST_SOURCES)
ALL_HEADERS := include/libpy/**.h

TEST_DEFINES = -DLIBPY_COMPILING_FOR_TESTS

UNSAFE_API ?= 1
ifneq ($(UNSAFE_API),0)
	TEST_DEFINES += -DLIBPY_AUTOCLASS_UNSAFE_API
endif

ALL_FLAGS := 'CC=$(CC) CXX=$(CXX) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) LDFLAGS=$(LDFLAGS)'

.PHONY: all
all: libpy/libpy.so

# Empty rule that should always trigger a build
.make/force:

# Write our current compiler flags so that we rebuild if they change.
ALL_FLAGS_MATCH := $(shell echo '$(ALL_FLAGS)' | cmp -s - .make/all-flags || echo 0)
ifeq ($(ALL_FLAGS_MATCH),0)
	ALL_FLAGS_DEPS := .make/force
endif
.make/all-flags: $(ALL_FLAGS_DEPS)
	@mkdir -p .make
	@echo '$(ALL_FLAGS)' > $@

$(SONAME): $(OBJECTS)
	$(CXX) $(OBJECTS) -shared -Wl,-$(SONAME_FLAG),$(SONAME_PATH) \
		-o $@ $(LDFLAGS)

$(SHORT_SONAME): $(SONAME)
	@rm -f $@
	ln -s $< $@

libpy/libpy.so: $(SHORT_SONAME)
	@rm -f $@
	ln -s ../$< $@

src/%.o: src/%.cc .make/all-flags
	$(CXX) $(CXXFLAGS) $(INCLUDE) -MD -fPIC -c $< -o $@

.PHONY: test
test: $(PYTHON_TESTS) $(TEST_MODULE)
	GTEST_OUTPUT=$(GTEST_OUTPUT) \
		$(LD_PRELOAD_VAR)="$(TEST_LD_PRELOAD)" \
		ASAN_OPTIONS=$(ASAN_OPTIONS) \
		LSAN_OPTIONS=$(LSAN_OPTIONS) \
		LSAN_OPTIONS=$(LSAN_OPTIONS) \
		GTEST_ARGS=--gtest_filter=$(GTEST_FILTER) \
		$(PYTEST) tests/ $(PYTEST_ARGS)

.PHONY: gdbtest
gdbtest: $(PYTHON_TESTS)
	@LD_LIBRARY_PATH=. GTEST_BREAK_ON_FAILURE=$(GTEST_BREAK) \
		gdb -ex run $(PYTEST) tests/

tests/%.o: tests/%.cc .make/all-flags
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(TEST_INCLUDE) $(TEST_DEFINES) \
		-isystem submodules/googletest/googletest/include \
		-isystem submodules/googletest/googletest/src \
		-MD -fPIC -c $< -o $@

$(TEST_MODULE): gtest.a $(TEST_OBJECTS) $(SONAME)
	$(CXX) -shared -o $@ $(TEST_OBJECTS) gtest.a $(TEST_INCLUDE) \
		-Wl,-rpath,`pwd` -lpthread -L. $(SONAME) $(LDFLAGS)

gtest.o: $(GTEST_SRCS) .make/all-flags
	$(CXX) $(filter-out $(WARNINGS),$(CXXFLAGS)) -I $(GTEST_DIR) \
	-I $(GTEST_DIR)/include -c $(GTEST_DIR)/src/gtest-all.cc -fPIC -o $@

gtest.a: gtest.o
	$(AR) $(ARFLAGS) $@ $^

.PHONY: tidy
tidy:
	$(CLANG_TIDY) $(ALL_SOURCES) $(ALL_HEADERS) --header-filter=include/ \
		-checks=-*,clang-analyzer-*,clang-analyzer-* \
		-- -x c++ --std=gnu++17 \
		$(INCLUDE) $(TEST_INCLUDE) $(shell)

.PHONY: format
format:
	@$(CLANG_FORMAT) -i $(ALL_SOURCES) $(ALL_HEADERS)


.PHONY: clean
clean:
	@rm -f $(SONAME) $(SHORT_SONAME) $(OBJECTS) $(DFILES) \
		$(TEST_MODULE) $(TEST_OBJECTS) $(TEST_DFILES) \
		gtest.o gtest.a

-include $(DFILES) $(TEST_DFILES)

print-%:
	@echo $* = $($*)
