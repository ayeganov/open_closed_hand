from conans import ConanFile, CMake


class Gateway(ConanFile):
    name = "Gateway"
    settings = "os", "compiler", "build_type", "arch"
    exports = "*"
    build_policy = "missing"
    requires = (
        "opencv/4.3.0@conan/stable"
    )
    generators = "cmake"

    def build(self):
        cmake = CMake(self)
        self.run(f"cmake --build . {cmake.build_config} -j8")
