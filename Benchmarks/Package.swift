// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "benchmarks",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .watchOS(.v11),
        .tvOS(.v18),
        .visionOS(.v2),
    ],
    dependencies: [
        .package(name: "swift-gguf", path: "../"),
        .package(url: "https://github.com/ordo-one/package-benchmark", from: "1.29.7"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.5.1"),
    ],
    targets: [
        .executableTarget(
            name: "GGMLBenchmarks",
            dependencies: [
                .product(name: "Benchmark", package: "package-benchmark"),
                .product(name: "GGUF", package: "swift-gguf"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Benchmarks",
            resources: [
                .copy("Data")
            ],
            plugins: [
                .plugin(name: "BenchmarkPlugin", package: "package-benchmark")
            ]
        )
    ]
)
