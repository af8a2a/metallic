#include <filesystem>
#include <fstream>

int wmain(int argc, wchar_t** argv)
{
    if (argc != 2) {
        return 1;
    }

    const std::filesystem::path capturePath(argv[1]);
    auto receiptPath = capturePath;
    receiptPath += L".opened";
    const auto captureUtf8 = capturePath.u8string();
    std::ofstream receipt(receiptPath, std::ios::binary);
    receipt.write(reinterpret_cast<const char*>(captureUtf8.data()),
        static_cast<std::streamsize>(captureUtf8.size()));
    return receipt ? 0 : 2;
}
