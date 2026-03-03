#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
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
	uint16_t vertices_per_frame; // important, because this is number of verts per frame :/
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
	const std::span buffer(reinterpret_cast<uint8_t*>(&result), sizeof(T));
	const size_t num_bytes_read = read(file, buffer).size_bytes();

	if (num_bytes_read != sizeof(T))
	{
		throw std::runtime_error("failed to read sufficient number of bytes for given type");
	}

	return result;
}

std::optional<RSDKModel> load_model(std::ifstream& file)
{
	RSDKModel model {};

	std::array<char, 4> fourcc {};
	static_assert(sizeof(char) == sizeof(uint8_t));
	std::span<uint8_t> bytes_read = read(file, std::span(reinterpret_cast<uint8_t*>(fourcc.data()), fourcc.size()));

	if (bytes_read.size_bytes() < fourcc.size() || memcmp(fourcc.data(), "MDL\0", fourcc.size()) != 0)
	{
		std::cerr << "not a valid RSDK model" << std::endl;
		return std::nullopt;
	}

	model.flags = read_t<uint8_t>(file);

	if (model.flags & RSDKModelFlags::use_textures)
	{
		std::cerr << "textured models not currently supported" << std::endl;
		return std::nullopt;
	}

	if (model.flags & (RSDKModelFlags::is_stripped | RSDKModelFlags::is_baked))
	{
		std::cerr << "model is already optimized" << std::endl;
		return std::nullopt;
	}

	model.face_vertex_count = read_t<uint8_t>(file);

	model.vertices_per_frame = read_t<uint16_t>(file);
	model.frame_count = read_t<uint16_t>(file);

	model.vertices.resize(model.vertices_per_frame * model.frame_count);

	if (model.flags & RSDKModelFlags::use_textures)
	{
		model.tex_coords.resize(model.vertices_per_frame);

		for (RSDKTexCoord& tex_coord : model.tex_coords)
		{
			static_assert(sizeof(float) == sizeof(uint32_t));
			tex_coord.x = read_t<float>(file);
			tex_coord.y = read_t<float>(file);
		}
	}

	if (model.flags & RSDKModelFlags::use_colors)
	{
		model.colors.resize(model.vertices_per_frame);

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
		for (uint16_t v = 0; v < model.vertices_per_frame; ++v)
		{
			const size_t i = (static_cast<size_t>(f) * model.vertices_per_frame) + v;

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

std::optional<RSDKModel> load_model(const std::string& path)
{
	std::ifstream input_file(path, std::ios::binary);

	if (!input_file.is_open())
	{
		std::cerr << "failed to open file: " << path << std::endl;
		return std::nullopt;
	}

	return load_model(input_file);
}

size_t write(std::ofstream& file, std::span<const uint8_t> buffer)
{
	const auto begin = file.tellp();
	file.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
	const auto end = file.tellp();

	return static_cast<size_t>(end - begin);
}

template <typename T>
void write_t(std::ofstream& file, const T& data)
{
	const std::span buffer(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
	const size_t num_bytes_written = write(file, buffer);

	if (num_bytes_written != sizeof(T))
	{
		throw std::runtime_error("failed to write sufficient number of bytes for given type");
	}
}

void write_model(std::ofstream& file, const RSDKModel& model)
{
	write_t(file, 'M');
	write_t(file, 'D');
	write_t(file, 'L');
	write_t(file, '\0');

	write_t(file, model.flags);
	write_t(file, model.face_vertex_count);
	write_t(file, model.vertices_per_frame);
	write_t(file, model.frame_count);

	if (model.flags & RSDKModelFlags::use_colors)
	{
		for (const RSDKColor& color : model.colors)
		{
			write_t(file, color.color);
		}
	}

	write_t(file, static_cast<uint16_t>(model.indices.size()));

	for (uint16_t index : model.indices)
	{
		write_t(file, index);
	}

	for (const RSDKModelVertex& vertex : model.vertices)
	{
		write_t(file, vertex.x);
		write_t(file, vertex.y);
		write_t(file, vertex.z);

		if (model.flags & RSDKModelFlags::use_normals)
		{
			write_t(file, vertex.nx);
			write_t(file, vertex.ny);
			write_t(file, vertex.nz);
		}
	}
}

bool write_model(const std::string& path, const RSDKModel& model)
{
	std::ofstream output_file(path, std::ios::binary | std::ios::trunc);

	if (!output_file.is_open())
	{
		std::cerr << "failed write file: " << path << std::endl;
		return false;
	}

	write_model(output_file, model);
	return true;
}

void get_verts_for_optimizer(const RSDKModel& model, size_t frame_number, std::span<VertexForOptimizer> out_verts)
{
	if (out_verts.size() < model.vertices_per_frame)
	{
		throw std::runtime_error("output buffer is too small for vertices");
	}

	const std::span in_verts(&model.vertices[model.vertices_per_frame * frame_number], model.vertices_per_frame);

	for (size_t i = 0; i < model.vertices_per_frame; ++i)
	{
		const RSDKModelVertex& old_vert = in_verts[i];
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

		out_verts[i] = new_vert;
	}
}

struct RemapInfo
{
	size_t new_vertex_count;
	std::vector<uint32_t> remap_indices;
};

[[nodiscard]] RemapInfo get_remap_info(std::span<const uint16_t> indices, std::span<const VertexForOptimizer> vertices)
{
	RemapInfo result {};
	result.remap_indices.resize(indices.size());

	result.new_vertex_count =
		meshopt_generateVertexRemap(result.remap_indices.data(),
		                            indices.data(),
		                            indices.size(),
		                            vertices.data(),
		                            vertices.size(),
		                            sizeof(VertexForOptimizer));

	return result;
}

void remap_vertices(const RemapInfo& remap_info, std::span<const VertexForOptimizer> in_verts, std::span<VertexForOptimizer> out_verts)
{
	if (out_verts.size() < remap_info.new_vertex_count)
	{
		throw std::runtime_error("output buffer is too small for vertices");
	}

	meshopt_remapVertexBuffer(out_verts.data(),
	                          in_verts.data(),
	                          in_verts.size(),
	                          sizeof(VertexForOptimizer),
	                          remap_info.remap_indices.data());
}

void remap_indices(const RemapInfo& remap_info, std::span<const uint16_t> in_indices, std::span<uint16_t> out_indices)
{
	if (out_indices.size() < in_indices.size())
	{
		throw std::runtime_error("output buffer is too small for indices");
	}

	meshopt_remapIndexBuffer(out_indices.data(),
	                         in_indices.data(),
	                         in_indices.size(),
	                         remap_info.remap_indices.data());
}

[[nodiscard]] std::vector<uint16_t> quad_to_tri_indices(std::span<const uint16_t> in_indices)
{
	std::vector<uint16_t> result;
	result.reserve((in_indices.size() / 4) * 6);

	for (size_t i = 0; i + 3 < in_indices.size(); ++i)
	{
		result.emplace_back(in_indices[i + 0]);
		result.emplace_back(in_indices[i + 1]);
		result.emplace_back(in_indices[i + 2]);

		result.emplace_back(in_indices[i + 0]);
		result.emplace_back(in_indices[i + 2]);
		result.emplace_back(in_indices[i + 3]);
	}

	return result;
}

[[nodiscard]] std::vector<uint16_t> stripify(const uint8_t verts_per_face, std::span<const uint16_t> in_indices, size_t vertex_count)
{
	std::vector<uint16_t> tri_indices;
	std::span<const uint16_t> index_view;

	if (verts_per_face == 4)
	{
		tri_indices = quad_to_tri_indices(in_indices);
		index_view = tri_indices;
	}
	else
	{
		index_view = in_indices;
	}

	std::vector<uint16_t> strip_indices(meshopt_stripifyBound(index_view.size()));

	const size_t strip_index_count =
		meshopt_stripify(strip_indices.data(),
		                 index_view.data(),
		                 index_view.size(),
		                 vertex_count,
		                 static_cast<uint16_t>(0xFFFF));

	strip_indices.resize(strip_index_count);
	strip_indices.shrink_to_fit();
	return strip_indices;
}

int main(int argc, char** argv)
{
	std::string input_path;
	std::string output_path;

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

		if (arg == "-o" || arg == "--output")
		{
			if (i + 1 == argc)
			{
				break;
			}

			output_path = argv[++i];
			continue;
		}
	}

	if (input_path.empty())
	{
		std::cerr << "no input file specified. use -i or --input" << std::endl;
		return -1;
	}

	if (output_path.empty())
	{
		std::cerr << "no output file specified. use -o or --output" << std::endl;
		return -1;
	}

	std::cout << "loading file: " << input_path << std::endl << std::endl;

	const std::optional<RSDKModel> model_maybe = load_model(input_path);

	if (!model_maybe.has_value())
	{
		return -1;
	}

	const RSDKModel& model = model_maybe.value();

	std::cout
		<< std::format("input model stats:\n"
		               "\t verts per face: {}\n"
		               "\tverts per frame: {}\n"
		               "\t    frame count: {}\n"
		               "\t    total verts: {}\n"
		               "\tindices (faces): {} ({})\n",
		               static_cast<uint16_t>(model.face_vertex_count),
		               model.vertices_per_frame,
		               model.frame_count,
		               model.vertices.size(),
		               model.indices.size(),
		               (model.indices.size() / model.face_vertex_count))
		<< std::endl;

	std::cout << "optimizing..." << std::endl << std::endl;

	std::vector<VertexForOptimizer> vertices(model.vertices_per_frame * static_cast<size_t>(model.frame_count));
	std::vector<uint16_t> indices;

	for (size_t f = 0; f < model.frame_count; ++f)
	{
		std::span out_frame_verts(&vertices[model.vertices_per_frame * f], model.vertices_per_frame);
		get_verts_for_optimizer(model, f, out_frame_verts);
	}

	size_t new_vertex_count;

	{
		const RemapInfo remap_info = get_remap_info(model.indices, std::span(vertices.data(), model.vertices_per_frame));

		indices.resize(model.indices.size());
		remap_indices(remap_info, model.indices, indices);

		std::vector<VertexForOptimizer> new_verts(remap_info.new_vertex_count * model.frame_count);

		for (size_t f = 0; f < model.frame_count; ++f)
		{
			const std::span in_verts(&vertices[model.vertices_per_frame * f], model.vertices_per_frame);
			const std::span out_verts(&new_verts[remap_info.new_vertex_count * f], remap_info.new_vertex_count);

			remap_vertices(remap_info, in_verts, out_verts);
		}

		vertices = std::move(new_verts);
		new_vertex_count = remap_info.new_vertex_count;
	}

	// TODO: if (simplify)
	{
		constexpr float threshold = 0.2f;
		constexpr float target_error = 0.01f; // docs use 0.01f (1e-2f; <= 1%) error

		const auto target_index_count = static_cast<size_t>(static_cast<float>(indices.size()) * threshold);

		std::vector<uint16_t> lod_indices(indices.size());
		float lod_error = 0.0f; // <- reported back from simplify func

		// normal x, y, z, followed by color.
		// these weights might need adjusting...
		std::array<float, 7> attribute_weights = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

		const std::span frame_vertices(&vertices[0], new_vertex_count);

		// if necessary, texture coordinates could be included; they just need to be immediately
		// adjacent to the float colors.
		const size_t new_index_count =
			meshopt_simplifyWithAttributes(lod_indices.data(),
			                               indices.data(),
			                               indices.size(),
			                               &frame_vertices[0].vertex.x,
			                               frame_vertices.size(),
			                               sizeof(VertexForOptimizer),
			                               &frame_vertices[0].vertex.nx,
			                               sizeof(VertexForOptimizer),
			                               attribute_weights.data(),
			                               attribute_weights.size(),
			                               nullptr,
			                               target_index_count,
			                               target_error,
			                               /* options */ 0,
			                               &lod_error);

		std::cout << "error rate: " << lod_error << std::endl;

		lod_indices.resize(new_index_count);
		lod_indices.shrink_to_fit();

		indices = std::move(lod_indices);
	}

	{
		const RemapInfo remap_info = get_remap_info(indices, std::span(vertices.data(), new_vertex_count));

		{
			std::vector<uint16_t> temp_indices(indices.size());
			remap_indices(remap_info, indices, temp_indices);
			indices = std::move(temp_indices);
		}

		std::vector<VertexForOptimizer> new_verts(remap_info.new_vertex_count * model.frame_count);

		for (size_t f = 0; f < model.frame_count; ++f)
		{
			const std::span in_verts(&vertices[new_vertex_count * f], new_vertex_count);
			const std::span out_verts(&new_verts[remap_info.new_vertex_count * f], remap_info.new_vertex_count);

			remap_vertices(remap_info, in_verts, out_verts);
		}

		vertices = std::move(new_verts);
		new_vertex_count = remap_info.new_vertex_count;
	}

	// TODO: if (stripify)
	{
		// seems weird that this operates on a triangle list but whatever...
		meshopt_optimizeVertexCacheStrip(indices.data(), indices.data(), indices.size(), new_vertex_count);
	}

	// TODO: if (stripify)
	{
		std::vector<uint16_t> strip_indices = stripify(model.face_vertex_count, indices, new_vertex_count);
		size_t total_strips = 0;
		size_t total_actual_strips = 0;
		size_t total_loose_tris = 0;
		size_t strip_length = 0;
		size_t longest_strip = 0;

		for (uint16_t index : strip_indices)
		{
			if (index != 0xFFFF)
			{
				++strip_length;
				continue;
			}

			++total_strips;

			if (strip_length > 3)
			{
				++total_actual_strips;
			}
			else
			{
				++total_loose_tris;
			}

			longest_strip = std::max(longest_strip, strip_length);
			strip_length = 0;
		}

		std::cout
			<< std::format("strip stats:\n"
			               "\t       strips: {}\n"
			               "\tactual strips: {}\n"
			               "\t   loose tris: {}\n"
			               "\tlongest strip: {}\n"
			               "\tstrip indices: {} (- ends: {}; vs {})\n",
			               total_strips,
			               total_actual_strips,
			               total_loose_tris,
			               longest_strip,
			               strip_indices.size(),
			               strip_indices.size() - total_strips,
			               indices.size())
			<< std::endl;

		// TODO: write strips to file
	}

	{
		const size_t vert_frame_delta = static_cast<size_t>(model.vertices_per_frame) - new_vertex_count;
		const size_t vertex_delta = model.vertices.size() - vertices.size();
		const size_t index_delta = model.indices.size() - indices.size();

		if (!vert_frame_delta && !vertex_delta && !index_delta)
		{
			std::cout << "no changes were made; not writing new file." << std::endl;
			return 0;
		}

		std::cout
			<< std::format("optimization stats:\n"
			               "\tverts per frame: {} -> {} (-{})\n"
			               "\t    total verts: {} -> {} (-{})\n"
			               "\t    index count: {} -> {} (-{})\n"
			               "\t     face count: {} -> {} (-{})\n",
			               model.vertices_per_frame, new_vertex_count, vert_frame_delta,
			               model.vertices.size(), vertices.size(), vertex_delta,
			               model.indices.size(), indices.size(), index_delta,
			               (model.indices.size() / model.face_vertex_count), (indices.size() / model.face_vertex_count), (index_delta / model.face_vertex_count))
			<< std::endl;
	}

	std::cout << "writing to file: " << output_path << std::endl;

	RSDKModel new_model {};
	new_model.flags = model.flags;
	new_model.face_vertex_count = model.face_vertex_count;
	new_model.vertices_per_frame = static_cast<uint16_t>(new_vertex_count);
	new_model.frame_count = model.frame_count;
	new_model.vertices.resize(vertices.size());
	new_model.indices = std::move(indices);

	// TODO: texture coordinates? currently ignored
	if (new_model.flags & RSDKModelFlags::use_colors)
	{
		new_model.colors.resize(new_vertex_count);
	}

	for (size_t i = 0; i < vertices.size(); ++i)
	{
		const VertexForOptimizer& old_vert = vertices[i];
		new_model.vertices[i] = old_vert.vertex;
	}

	if (new_model.flags & RSDKModelFlags::use_colors)
	{
		for (size_t i = 0; i < new_vertex_count; ++i)
		{
			const VertexForOptimizer& old_vert = vertices[i];

			auto& new_color = new_model.colors[i];

			for (size_t c = 0; c < old_vert.color.size(); ++c)
			{
				new_color.bytes[c] = static_cast<uint8_t>(old_vert.color[c] * 255.0f);
			}
		}
	}

	if (!write_model(output_path, new_model))
	{
		return -1;
	}

	return 0;
}
