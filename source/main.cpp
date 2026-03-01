#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include <meshoptimizer.h>

struct RSDKModelFlags
{
	enum : uint8_t
	{
		none = 0,
		use_normals = 1 << 0,
		use_textures = 1 << 1,
		use_colors = 1 << 2,

		//
		// KOS-specific extensions below
		//

		is_stripped = 1 << 3,
		is_baked = 1 << 4,
	};
};

struct RSDKModelVertex
{
	float x, y, z;
	float nx, ny, nz;
};

struct RSDKTexCoord
{
	float x, y;
};

union RSDKColor
{
	uint8_t bytes[sizeof(uint32_t)];
	uint32_t color;
};

struct RSDKModel
{
	uint8_t flags;
	uint8_t face_vertex_count; // verts per face
	uint16_t vertex_count; // important, because this is number of verts per frame :/
	uint16_t frame_count;

	std::vector<RSDKModelVertex> vertices;
	std::vector<RSDKTexCoord> tex_coords;
	std::vector<RSDKColor> colors;
	std::vector<uint16_t> indices;

	//
	// KOS-specific extensions
	//

	uint16_t strip_count;
	uint16_t loose_tri_count;
};

[[nodiscard]] bool equals_case_insensitive(const std::string_view& a, const std::string_view& b)
{
	auto fn = [](unsigned char ca, unsigned char cb) { return std::tolower(ca) == std::tolower(cb); };
	return std::ranges::equal(a, b, fn);
}

[[nodiscard]] std::span<uint8_t> read(std::ifstream& file, std::span<uint8_t> buffer)
{
	const auto begin = file.tellg();
	file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
	const auto end = file.tellg();

	const auto num_bytes_read = static_cast<size_t>(end - begin);
	return buffer.subspan(0, num_bytes_read);
}

template <typename T>
[[nodiscard]] T read_t(std::ifstream& file)
{
	T result;
	const size_t num_bytes_read = read(file, std::span(reinterpret_cast<uint8_t*>(&result), sizeof(T))).size_bytes();

	if (num_bytes_read != sizeof(T))
	{
		// WIP
		throw;
	}

	return result;
}

RSDKModel load_model(std::ifstream& file)
{
	RSDKModel model {};

	std::array<char, 4> fourcc {};
	static_assert(sizeof(char) == sizeof(uint8_t));
	std::span<uint8_t> bytes_read = read(file, std::span(reinterpret_cast<uint8_t*>(fourcc.data()), fourcc.size()));

	if (bytes_read.size_bytes() < fourcc.size() || memcmp(fourcc.data(), "MDL\0", fourcc.size()) != 0)
	{
		std::cerr << "not a valid RSDK model" << std::endl;
		return model;
	}

	model.flags = read_t<uint8_t>(file);
	model.face_vertex_count = read_t<uint8_t>(file);

	model.vertex_count = read_t<uint16_t>(file);
	model.frame_count = read_t<uint16_t>(file);

	model.vertices.resize(model.vertex_count * model.frame_count);

	if (model.flags & RSDKModelFlags::use_textures)
	{
		model.tex_coords.resize(model.vertex_count);

		for (RSDKTexCoord& tex_coord : model.tex_coords)
		{
			static_assert(sizeof(float) == sizeof(uint32_t));
			tex_coord.x = read_t<float>(file);
			tex_coord.y = read_t<float>(file);
		}
	}

	if (model.flags & RSDKModelFlags::use_colors)
	{
		model.colors.resize(model.vertex_count);

		for (RSDKColor& color : model.colors)
		{
			color.color = read_t<uint32_t>(file);
		}
	}

	// ignoring KOS-specific extensions for now

	const auto index_count = read_t<uint16_t>(file);
	model.indices.resize(index_count);

	for (uint16_t& index : model.indices)
	{
		index = read_t<uint16_t>(file);
	}

	for (uint16_t f = 0; f < model.frame_count; ++f)
	{
		for (uint16_t v = 0; v < model.vertex_count; ++v)
		{
			const size_t i = (static_cast<size_t>(f) * model.vertex_count) + v;

			RSDKModelVertex& vertex = model.vertices[i];

			vertex.x = read_t<float>(file);
			vertex.y = read_t<float>(file);
			vertex.z = read_t<float>(file);

			if (model.flags & RSDKModelFlags::use_normals)
			{
				vertex.nx = read_t<float>(file);
				vertex.ny = read_t<float>(file);
				vertex.nz = read_t<float>(file);
			}
			else
			{
				vertex.nx = 0.0f;
				vertex.ny = 0.0f;
				vertex.nz = 0.0f;
			}
		}
	}

	return model;
}

int main(int argc, char** argv)
{
	std::string input_path;

	for (int i = 1; i < argc; ++i)
	{
		const std::string_view arg(argv[i]);

		if (arg == "-i" || arg == "--input")
		{
			if (i + 1 == argc)
			{
				break;
			}

			input_path = argv[++i];
			continue;
		}
	}

	if (input_path.empty())
	{
		std::cerr << "no input file specified. use -i or --input" << std::endl;
		return -1;
	}

	std::ifstream file(input_path, std::ios::binary);

	if (!file.is_open())
	{
		std::cerr << "failed to open file: " << input_path << std::endl;
		return -2;
	}

	RSDKModel model = load_model(file);

	std::cout
		<< " verts per face: " << static_cast<uint16_t>(model.face_vertex_count) << std::endl
		<< "verts per frame: " << model.vertex_count << std::endl
		<< "    frame count: " << model.frame_count << std::endl
		<< "    total verts: " << model.vertices.size() << std::endl
		<< " indices (faces): " << model.indices.size() << " (" << (model.indices.size() / model.face_vertex_count) << ')' << std::endl;

	// TODO: optimize mesh

	return 0;
}
