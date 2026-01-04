# `swift-gguf`

A parser for the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format.
Done using [swift-binary-parsing](https://github.com/apple/swift-binary-parsing).

## Usage

```swift
let fileData = try Data(contentsOf: url, options: .mappedIfSafe)
let gguf = try GGUF(parsing: fileData)
print("Version: \(gguf.header.version)")
print("Tensors: \(gguf.tensorInfos.count)")

// Access metadata
if let modelName = gguf.metadataValue(forKey: "general.name") {
    print("Model: \(modelName)")
}

// Load float array
let tensor = try gguf.tensorFloatArray(at: 0, from: fileData)
```

## Acknowledgements

This project uses some of the code from:

- [ggml](https://github.com/ggml-org/ggml)

## Code Formatting

This project uses [swift-format](https://github.com/swiftlang/swift-format). To format the code run:

```bash
swift format . -i -r --configuration .swift-format
```
