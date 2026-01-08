// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
import PackageDescription

let package = Package(
    name: "swift-gguf",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .watchOS(.v11),
        .tvOS(.v18),
        .visionOS(.v2),
    ],
    products: [
        .library(
            name: "GGUF",
            targets: ["GGUF"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-binary-parsing",
            revision: "3954d6fa395881629ad2ca4f42928766ededc1cc"
        ),
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
        .target(
            name: "TestData",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics")
            ],
            resources: [
                .copy("Data")
            ]
        ),
        .testTarget(
            name: "DequantizeTests",
            dependencies: [
                "Dequantize",
                "TestData",
                .product(name: "Numerics", package: "swift-numerics"),
            ]
        ),
        .testTarget(
            name: "GGUFTests",
            dependencies: [
                "Dequantize",
                "GGUF",
                "TestData",
                .product(name: "Numerics", package: "swift-numerics"),
            ]
        ),
    ]
)
