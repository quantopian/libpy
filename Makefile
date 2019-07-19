# Put custom environment stuff here.
-include Makefile.local

PYTHON ?= python

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
	-fmax-errors=$(MAX_ERRORS) $(WARNINGS) \
	-DPY_MAJOR_VERSION=$(PY_MAJOR_VERSION) \
	-DPY_MINOR_VERSION=$(PY_MINOR_VERSION)

INCLUDE_DIRS := include/
INCLUDE := $(foreach d,$(INCLUDE_DIRS), -I$d) \
	$(shell $(PYTHON)-config --includes) \
	-I $(shell $(PYTHON) -c 'import numpy as np;print(np.get_include())')

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
TESTRUNNER := tests/runner.py

ALL_SOURCES := $(SOURCES) $(TEST_SOURCES)
ALL_HEADERS := include/libpy/**.h

TEST_DEFINES = -DLIBPY_COMPILING_FOR_TESTS

UNSAFE_API ?= 1
ifneq ($(UNSAFE_API),0)
	TEST_DEFINES += -DLIBPY_AUTOCLASS_UNSAFE_API
endif

ALL_FLAGS := 'CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) LDFLAGS=$(LDFLAGS)'

.PHONY: all
all: $(SHORT_SONAME)

.PHONY: local-install
local-install: $(SONAME)
	cp $< ~/lib
	@rm -f ~/lib/$(SHORT_SONAME)
	ln -s ~/lib/$(SONAME) ~/lib/$(SHORT_SONAME)
	cp -rf include/$(LIBRARY) ~/include

# Write our current compiler flags so that we rebuild if they change.
force:
.compiler_flags: force
	@echo '$(ALL_FLAGS)' | cmp -s - $@ || echo '$(ALL_FLAGS)' > $@

$(SONAME): $(OBJECTS)
	$(CXX) $(OBJECTS) -shared -Wl,-$(SONAME_FLAG),$(SONAME_PATH) \
		-o $@ $(LDFLAGS)
	@rm -f $(SHORT_SONAME)

$(SHORT_SONAME): $(SONAME)
	ln -s $(SONAME) $(SHORT_SONAME)

src/%.o: src/%.cc .compiler_flags
	$(CXX) $(CXXFLAGS) $(INCLUDE) -MD -fPIC -c $< -o $@

.PHONY: test
test: $(TESTRUNNER) $(TEST_MODULE)
	GTEST_OUTPUT=$(GTEST_OUTPUT) \
		$(LD_PRELOAD_VAR)="$(TEST_LD_PRELOAD)" \
		ASAN_OPTIONS=$(ASAN_OPTIONS) \
		LSAN_OPTIONS=$(LSAN_OPTIONS) \
		LSAN_OPTIONS=$(LSAN_OPTIONS) \
		PYTHONPATH=tests/ \
		$(PYTHON) $< --gtest_filter=$(GTEST_FILTER)

.PHONY: gdbtest
gdbtest: $(TESTRUNNER)
	@LD_LIBRARY_PATH=. GTEST_BREAK_ON_FAILURE=$(GTEST_BREAK) gdb -ex run $<

tests/%.o: tests/%.cc .compiler_flags
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(TEST_INCLUDE) $(TEST_DEFINES) \
		-isystem submodules/googletest/googletest/include \
		-isystem submodules/googletest/googletest/src \
		-MD -fPIC -c $< -o $@

$(TEST_MODULE): gtest.a $(TEST_OBJECTS) $(SONAME)
	$(CXX) -shared -o $@ $(TEST_OBJECTS) gtest.a $(TEST_INCLUDE) \
		-Wl,-rpath,`pwd` -lpthread -L. $(SONAME) $(LDFLAGS)

gtest.o: $(GTEST_SRCS) .compiler_flags
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
