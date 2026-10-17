CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
CPPFLAGS ?=
LDFLAGS ?=
PYTHON ?= python

SOURCES := src/Punto3.cpp src/main_sin_modularizar.cpp
HEADERS := include/rayleigh.h

ifeq ($(OS),Windows_NT)
TARGET := rayleigh.exe
REMOVE_FILE := cmd.exe /C del /Q
else
TARGET := rayleigh
REMOVE_FILE := rm -f
endif

.PHONY: all run plot ensemble clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Iinclude $(SOURCES) $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

plot: data/simulacion.csv
	$(PYTHON) scripts/graficar_rayleigh.py

ensemble:
	$(PYTHON) scripts/analizar_ensamble_rayleigh.py

clean:
	$(REMOVE_FILE) $(TARGET)
