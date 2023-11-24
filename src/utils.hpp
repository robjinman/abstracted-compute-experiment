#pragma once

#include <string>

#define STR(x) (std::stringstream("") << x).str()

void trimLeft(std::string& s);
void trimRight(std::string& s);
