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

struct VertexForOptimizer
{
	RSDKModelVertex vertex;
	// color as floats. I don't care which ones are which color channels.
	std::array<float, 4> color;
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

	std::cout << "checking file: " << input_path << std::endl << std::endl;

	RSDKModel model = load_model(file);

	std::cout
		<< " verts per face: " << static_cast<uint16_t>(model.face_vertex_count) << std::endl
		<< "verts per frame: " << model.vertex_count << std::endl
		<< "    frame count: " << model.frame_count << std::endl
		<< "    total verts: " << model.vertices.size() << std::endl
		<< " indices (faces): " << model.indices.size() << " (" << (model.indices.size() / model.face_vertex_count) << ')' << std::endl
		<< std::endl;

	std::vector<VertexForOptimizer> vertices(model.vertex_count);
	std::vector<uint16_t> indices;

	// copy the first frame of vertices into a contiguous structure so that
	// meshoptimizer is aware of all components for remapping.
	for (size_t i = 0; i < vertices.size(); ++i)
	{
		const auto& old_vert = model.vertices[i];
		RSDKColor color;

		if ((model.flags & RSDKModelFlags::use_colors))
		{
			color = model.colors[i];
		}
		else
		{
			color = {};
		}

		// meshoptmizer suggests memsetting the structure in case there's gaps
		// because it does byte-wise comparisons.
		VertexForOptimizer new_vert;
		memset(&new_vert, 0, sizeof(VertexForOptimizer));
		new_vert.vertex = old_vert;
		new_vert.color[0] = static_cast<float>(color.bytes[0]) / 255.0f;
		new_vert.color[1] = static_cast<float>(color.bytes[1]) / 255.0f;
		new_vert.color[2] = static_cast<float>(color.bytes[2]) / 255.0f;
		new_vert.color[3] = static_cast<float>(color.bytes[3]) / 255.0f;

		vertices[i] = new_vert;
	}

	{
		std::vector<uint32_t> remap_indices(model.indices.size());

		const size_t remapped_vert_count =
			meshopt_generateVertexRemap(remap_indices.data(),
			                            model.indices.data(),
			                            model.indices.size(),
			                            vertices.data(),
			                            vertices.size(),
			                            sizeof(VertexForOptimizer));

		std::vector<uint16_t> new_indices(model.indices.size());
		std::vector<VertexForOptimizer> new_vertices(remapped_vert_count);

		meshopt_remapIndexBuffer(new_indices.data(),
		                         model.indices.data(),
		                         model.indices.size(),
		                         remap_indices.data());

		meshopt_remapVertexBuffer(new_vertices.data(),
		                          vertices.data(),
		                          vertices.size(),
		                          sizeof(VertexForOptimizer),
		                          remap_indices.data());

		indices = std::move(new_indices);
		vertices = std::move(new_vertices);
	}

	{
		constexpr float threshold = 0.2f;
		constexpr float target_error = 0.01f; // docs use 0.01f (<= 1%) error

		const auto target_index_count = static_cast<size_t>(static_cast<float>(indices.size()) * threshold);

		std::vector<uint16_t> lod_indices(indices.size());
		float lod_error = 0.0f; // <- reported back from simplify func

		decltype(VertexForOptimizer::color) attribute_weights = { 1.0f, 1.0f, 1.0f, 1.0f };

		// if necessary, texture coordinates could be included; they just need to be immediately
		// adjacent to the float colors.
		const size_t new_index_count =
			meshopt_simplifyWithAttributes(lod_indices.data(),
			                               indices.data(),
			                               indices.size(),
			                               &vertices[0].vertex.x,
			                               vertices.size(),
			                               sizeof(VertexForOptimizer),
			                               vertices[0].color.data(),
			                               sizeof(VertexForOptimizer),
			                               attribute_weights.data(),
			                               attribute_weights.size(),
			                               nullptr,
			                               target_index_count,
			                               target_error,
			                               /* options */ 0,
			                               &lod_error);

		const size_t delta = indices.size() - new_index_count;

		std::cout
			<< "index count change: " << indices.size() << " -> " << new_index_count
			<< " (" << delta << ')' << std::endl
			<< " face count change: " << indices.size() / model.face_vertex_count << " -> " << new_index_count / model.face_vertex_count
			<< " (" << delta / model.face_vertex_count << ')' << std::endl
			<< std::endl;

		lod_indices.resize(new_index_count);
		lod_indices.shrink_to_fit();
	}

	// TODO: output to file

	return 0;
}
