// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

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
            name: "GGMLQuants",
            dependencies: []
        ),
        .target(
            name: "Quants",
            dependencies: [
                "GGMLQuants"
            ]
        ),
        .target(
            name: "GGUF",
            dependencies: [
                "Quants",
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
            name: "QuantsTests",
            dependencies: [
                "Quants",
                "TestData",
                .product(name: "Numerics", package: "swift-numerics"),
            ]
        ),
        .testTarget(
            name: "GGUFTests",
            dependencies: [
                "Quants",
                "GGUF",
                "TestData",
                .product(name: "Numerics", package: "swift-numerics"),
            ]
        ),
    ]
)
