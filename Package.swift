// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-gguf",
    platforms: [
        .macOS(.v15), .iOS(.v18), .watchOS(.v11), .tvOS(.v18), .visionOS(.v2),
    ],
    products: [
        .library(
            name: "GGUF",
            targets: ["GGUF"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-binary-parsing", from: "0.0.1"),
        .package(url: "https://github.com/apple/swift-numerics.git", from: "1.1.1"),
    ],
    targets: [
        .target(
            name: "GGMLDequantize",
            dependencies: []
        ),
        .target(
            name: "Dequantize",
            dependencies: [
                "GGMLDequantize"
            ]
        ),
        .target(
            name: "GGUF",
            dependencies: [
                "Dequantize",
                .product(name: "BinaryParsing", package: "swift-binary-parsing"),
            ]
        ),
        .testTarget(
            name: "GGUFTests",
            dependencies: [
                "GGUF",
                "Dequantize",
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            resources: [
                .copy("Resources")
            ]
        ),
    ]
)
