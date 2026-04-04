#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <meshoptimizer.h>

struct Vector3
{
	float x, y, z;

	[[nodiscard]] float sqr_magnitude() const
	{
		return (x * x) + (y * y) + (z * z);
	}

	[[nodiscard]] float magnitude() const
	{
		return std::sqrt(sqr_magnitude());
	}

	[[nodiscard]] Vector3 normalized() const
	{
		const float mag = magnitude();

		if (mag >= std::numeric_limits<float>::epsilon())
		{
			return { x / mag, y / mag, z / mag };
		}

		return { 0.0f, 0.0f, 0.0f };
	}

	Vector3& operator+=(const Vector3& rhs)
	{
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
		return *this;
	}

	Vector3& operator/=(float rhs)
	{
		x /= rhs;
		y /= rhs;
		z /= rhs;
		return *this;
	}
};

struct Vector4
{
	float x, y, z, w;
};

struct RSDKModelFlags
{
	enum : uint8_t
	{
		none         = 0,
		use_normals  = 1 << 0,
		use_textures = 1 << 1,
		use_colors   = 1 << 2,

		//
		// KOS-specific extensions below
		//

		is_stripped = 1 << 3,
		is_baked    = 1 << 4,
	};
};

struct RSDKModelVertex
{
	Vector3 position;
	Vector3 normal;
};

struct RSDKTexCoord
{
	float u, v;
};

union RSDKColor
{
	uint8_t bytes[sizeof(uint32_t)];
	uint32_t u32;

	struct
	{
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	};
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

	std::vector<uint16_t> strip_lengths;
	std::vector<uint16_t> strip_indices;
	std::vector<uint16_t> loose_tri_indices;
};

struct VertexForOptimizer
{
	Vector3 position;
	Vector3 normal;
	RSDKTexCoord tex_coord;
	Vector4 color; // don't care which channels are which
};

[[nodiscard]] float dot(const Vector3& lhs, const Vector3& rhs)
{
	return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
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
			tex_coord.u = read_t<float>(file);
			tex_coord.v = read_t<float>(file);
		}
	}

	if (model.flags & RSDKModelFlags::use_colors)
	{
		model.colors.resize(model.vertices_per_frame);

		for (RSDKColor& color : model.colors)
		{
			color.u32 = read_t<uint32_t>(file);
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

			vertex.position.x = read_t<float>(file);
			vertex.position.y = read_t<float>(file);
			vertex.position.z = read_t<float>(file);

			if (model.flags & RSDKModelFlags::use_normals)
			{
				vertex.normal.x = read_t<float>(file);
				vertex.normal.y = read_t<float>(file);
				vertex.normal.z = read_t<float>(file);
			}
			else
			{
				vertex.normal = { 0.0f, 0.0f, 0.0f };
			}
		}
	}

	return model;
}

std::optional<RSDKModel> load_model(const std::filesystem::path& file_path)
{
	std::ifstream input_file(file_path, std::ios::binary);

	if (!input_file.is_open())
	{
		std::cerr << "failed to open file: " << file_path << std::endl;
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

	if (model.flags & RSDKModelFlags::use_textures)
	{
		for (const RSDKTexCoord& tex_coord : model.tex_coords)
		{
			write_t(file, tex_coord.u);
			write_t(file, tex_coord.v);
		}
	}

	if (model.flags & RSDKModelFlags::use_colors)
	{
		for (const RSDKColor& color : model.colors)
		{
			write_t(file, color.u32);
		}
	}

	if (model.flags & RSDKModelFlags::is_stripped)
	{
		// stripCount
		write_t(file, static_cast<uint16_t>(model.strip_lengths.size()));

		// looseTriCount
		write_t(file, static_cast<uint16_t>(model.loose_tri_indices.size() / 3));

		for (uint16_t strip_length : model.strip_lengths)
		{
			write_t(file, strip_length);
		}

		for (uint16_t strip_index : model.strip_indices)
		{
			write_t(file, strip_index);
		}

		for (uint16_t loose_tri_index : model.loose_tri_indices)
		{
			write_t(file, loose_tri_index);
		}
	}
	else
	{
		write_t(file, static_cast<uint16_t>(model.indices.size()));

		for (uint16_t index : model.indices)
		{
			write_t(file, index);
		}
	}

	for (const RSDKModelVertex& vertex : model.vertices)
	{
		write_t(file, vertex.position.x);
		write_t(file, vertex.position.y);
		write_t(file, vertex.position.z);

		if (model.flags & RSDKModelFlags::use_normals)
		{
			write_t(file, vertex.normal.x);
			write_t(file, vertex.normal.y);
			write_t(file, vertex.normal.z);
		}
	}
}

bool write_model(const std::filesystem::path& file_path, const RSDKModel& model)
{
	std::ofstream output_file(file_path, std::ios::binary | std::ios::trunc);

	if (!output_file.is_open())
	{
		std::cerr << "failed write file: " << file_path << std::endl;
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
		RSDKTexCoord tex_coord;
		RSDKColor color;

		if (model.flags & RSDKModelFlags::use_textures)
		{
			tex_coord = model.tex_coords[i];
		}
		else
		{
			tex_coord = { 0.0f, 0.0f };
		}

		if ((model.flags & RSDKModelFlags::use_colors))
		{
			color = model.colors[i];
		}
		else
		{
			color.u32 = 0;
		}

		// meshoptmizer suggests memsetting the structure in case there's gaps
		// because it does byte-wise comparisons.
		VertexForOptimizer new_vert;
		memset(&new_vert, 0, sizeof(VertexForOptimizer));

		new_vert.position = old_vert.position;
		new_vert.normal = old_vert.normal;
		new_vert.tex_coord = tex_coord;
		new_vert.color.x = static_cast<float>(color.bytes[0]) / 255.0f;
		new_vert.color.y = static_cast<float>(color.bytes[1]) / 255.0f;
		new_vert.color.z = static_cast<float>(color.bytes[2]) / 255.0f;
		new_vert.color.w = static_cast<float>(color.bytes[3]) / 255.0f;

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

	for (size_t i = 0; i + 3 < in_indices.size(); i += 4)
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

[[nodiscard]] std::vector<uint16_t> stripify(std::span<const uint16_t> in_indices, size_t vertex_count)
{
	std::vector<uint16_t> strip_indices(meshopt_stripifyBound(in_indices.size()));

	const size_t strip_index_count =
		meshopt_stripify(strip_indices.data(),
		                 in_indices.data(),
		                 in_indices.size(),
		                 vertex_count,
		                 static_cast<uint16_t>(0xFFFF));

	strip_indices.resize(strip_index_count);
	strip_indices.shrink_to_fit();
	return strip_indices;
}

bool bake_lighting(RSDKModel& model,
                   Vector3 light_direction,
                   float ambient_strength,
                   float diffuse_strength,
                   float specular_strength,
                   float specular_power)
{
	if (!(model.flags & RSDKModelFlags::use_normals))
	{
		return false;
	}

	model.flags |= RSDKModelFlags::is_baked;

	if (!(model.flags & RSDKModelFlags::use_colors))
	{
		model.flags |= RSDKModelFlags::use_colors;
		model.colors.resize(model.vertices_per_frame, { .u32 = 0xFFFFFFFF });
	}

	light_direction = light_direction.normalized();

	for (size_t i = 0; i < model.vertices_per_frame; ++i)
	{
		Vector3 average_normal = { 0.0f, 0.0f, 0.0f };

		// average normals across frames since vertex colors are shared between frames.
		for (size_t f = 0; f < model.frame_count; ++f)
		{
			const size_t start = f * static_cast<size_t>(model.vertices_per_frame);
			const RSDKModelVertex& vertex = model.vertices[start + i];
			average_normal += vertex.normal;
		}

		average_normal /= static_cast<float>(model.frame_count);
		average_normal = average_normal.normalized();

		const float ndotl = dot(average_normal, light_direction);

		constexpr float wrap = 0.15f;
		// const float diffuse = std::max(0.0f, ndotl) * diffuse_strength;
		const float diffuse_wrapped = std::max(0.0f, (ndotl + wrap) / (1.0f + wrap)) * diffuse_strength;
		const float specular = std::pow(std::max(0.0f, ndotl), specular_power) * specular_strength;
		const float brightness = ambient_strength + diffuse_wrapped + specular;

		RSDKColor& color = model.colors[i];
		color.b = static_cast<uint8_t>(std::clamp(static_cast<float>(color.b) * brightness, 0.0f, 255.0f));
		color.g = static_cast<uint8_t>(std::clamp(static_cast<float>(color.g) * brightness, 0.0f, 255.0f));
		color.r = static_cast<uint8_t>(std::clamp(static_cast<float>(color.r) * brightness, 0.0f, 255.0f));
	}

	return true;
}

struct Options
{
	std::filesystem::path input_file_path;
	std::filesystem::path output_file_path;

	bool create_output_dir = true;

	bool optimize = true;
	bool simplify = false;
	bool stripify = false;
	bool bake_lighting = false;

	uint16_t strip_max_points = 256;

	float simplify_index_threshold = 0.2f; // 0 for fewest indices possible
	float simplify_target_error = 0.01f; // <= 1%

	Vector3 bake_light_direction = { 0.15f, 0.85f, 0.1f };
	float bake_ambient_strength = 0.5f;
	float bake_diffuse_strength = 0.7f;
	float bake_specular_strength = 0.7f;
	float bake_specular_power = 1.5f;
};

[[nodiscard]] std::optional<Options> parse_args(std::span<char*> args);

int main(int argc, char** argv)
{
	const std::optional<Options> options_maybe = parse_args(std::span(argv, argc));

	if (!options_maybe.has_value())
	{
		return -1;
	}

	const Options& options = options_maybe.value();

	if (options.input_file_path.empty())
	{
		std::cerr << "no input file specified. use -i or --input" << std::endl;
		return -1;
	}

	if (options.output_file_path.empty())
	{
		std::cerr << "no output file specified. use -o or --output" << std::endl;
		return -1;
	}

	std::cout << "loading file: " << options.input_file_path << std::endl << std::endl;

	const std::optional<RSDKModel> model_maybe = load_model(options.input_file_path);

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

	if (model.face_vertex_count != 3 && model.face_vertex_count != 4)
	{
		std::cerr << "unsupported vertex-per-face count: " << static_cast<uint16_t>(model.face_vertex_count) << std::endl;
		return -1;
	}

	if (model.face_vertex_count == 4 && (options.simplify || options.stripify))
	{
		std::cout << "WARNING: quads will be converted to tris during simplification/strip generation! this will skew stat output!" << std::endl;
	}

	RSDKModel new_model {};
	new_model.flags = model.flags;
	new_model.face_vertex_count = model.face_vertex_count;
	new_model.vertices_per_frame = model.vertices_per_frame;
	new_model.frame_count = model.frame_count;
	new_model.indices = model.indices;

	std::vector<VertexForOptimizer> vertices(model.vertices_per_frame * static_cast<size_t>(model.frame_count));
	std::vector<uint16_t> quad_indices;

	for (size_t f = 0; f < model.frame_count; ++f)
	{
		std::span out_frame_verts(&vertices[model.vertices_per_frame * f], model.vertices_per_frame);
		get_verts_for_optimizer(model, f, out_frame_verts);
	}

	bool optimize_failed = !options.optimize;
	bool simplify_failed = !options.simplify;
	bool stripify_failed = !options.stripify;
	bool lighting_failed = !options.bake_lighting;

	if (options.optimize)
	{
		std::cout << "optimizing..." << std::endl;

		const RemapInfo remap_info = get_remap_info(model.indices, std::span(vertices.data(), model.vertices_per_frame));

		if (!remap_info.new_vertex_count || remap_info.new_vertex_count >= model.vertices_per_frame)
		{
			std::cout << "(failed; discarding changes)" << std::endl;
			optimize_failed = true;
		}
		else
		{
			remap_indices(remap_info, model.indices, new_model.indices);

			std::vector<VertexForOptimizer> new_verts(remap_info.new_vertex_count * model.frame_count);

			for (size_t f = 0; f < model.frame_count; ++f)
			{
				const std::span in_verts(&vertices[model.vertices_per_frame * f], model.vertices_per_frame);
				const std::span out_verts(&new_verts[remap_info.new_vertex_count * f], remap_info.new_vertex_count);

				remap_vertices(remap_info, in_verts, out_verts);
			}

			vertices = std::move(new_verts);
			new_model.vertices_per_frame = static_cast<uint16_t>(remap_info.new_vertex_count);
		}
	}

	if (new_model.face_vertex_count == 4 && (options.simplify || options.stripify))
	{
		quad_indices = std::exchange(new_model.indices, quad_to_tri_indices(new_model.indices));
		new_model.face_vertex_count = 3;
	}

	if (options.simplify)
	{
		std::cout << "simplifying..." << std::endl;

		std::vector<uint16_t> lod_indices(new_model.indices.size());
		const std::span frame_vertices(&vertices[0], new_model.vertices_per_frame);

		// these weights might need adjusting...
		// TODO: command-line args to override weights
		constexpr float k_normal_weight = 1.0f;
		constexpr float k_uv_weight     = 1.0f;
		constexpr float k_color_weight  = 1.0f;

		const float normal_weight = (model.flags & RSDKModelFlags::use_normals)  ? k_normal_weight : 0.0f;
		const float uv_weight     = (model.flags & RSDKModelFlags::use_textures) ? k_uv_weight     : 0.0f;
		const float color_weight  = (model.flags & RSDKModelFlags::use_colors)   ? k_color_weight  : 0.0f;

		const std::array<float, 9> attribute_weights =
		{
			// normal x, y, z
			normal_weight, normal_weight, normal_weight,
			// u, v
			uv_weight, uv_weight,
			// color x, y, z, w
			color_weight, color_weight, color_weight, color_weight
		};

		const auto target_index_count = static_cast<size_t>(static_cast<float>(new_model.indices.size()) * options.simplify_index_threshold);

		float lod_error = 0.0f; // <- reported back from simplify func

		const size_t new_index_count =
			meshopt_simplifyWithAttributes(lod_indices.data(),
			                               new_model.indices.data(),
			                               new_model.indices.size(),
			                               &frame_vertices[0].position.x,
			                               frame_vertices.size(),
			                               sizeof(VertexForOptimizer),
			                               &frame_vertices[0].normal.x,
			                               sizeof(VertexForOptimizer),
			                               attribute_weights.data(),
			                               attribute_weights.size(),
			                               nullptr,
			                               target_index_count,
			                               options.simplify_target_error,
			                               /* options */ 0,
			                               &lod_error);

		if (!new_index_count || new_index_count >= new_model.indices.size())
		{
			std::cout << "(failed; discarding changes)" << std::endl;
			simplify_failed = true;
		}
		else
		{
			lod_indices.resize(new_index_count);
			lod_indices.shrink_to_fit();

			std::cout << "error rate: " << lod_error << std::endl;

			const RemapInfo remap_info = get_remap_info(lod_indices, std::span(vertices.data(), new_model.vertices_per_frame));

			if (remap_info.new_vertex_count > 0 && remap_info.new_vertex_count < new_model.vertices_per_frame)
			{
				std::vector<uint16_t> temp_indices(lod_indices.size());
				remap_indices(remap_info, lod_indices, temp_indices);
				lod_indices = std::move(temp_indices);

				std::vector<VertexForOptimizer> new_verts(remap_info.new_vertex_count * model.frame_count);

				for (size_t f = 0; f < model.frame_count; ++f)
				{
					const std::span in_verts(&vertices[new_model.vertices_per_frame * f], new_model.vertices_per_frame);
					const std::span out_verts(&new_verts[remap_info.new_vertex_count * f], remap_info.new_vertex_count);

					remap_vertices(remap_info, in_verts, out_verts);
				}

				vertices = std::move(new_verts);
				new_model.vertices_per_frame = remap_info.new_vertex_count;
			}

			new_model.indices = std::move(lod_indices);
		}
	}

	if (options.optimize || options.simplify)
	{
		const auto vert_frame_delta = static_cast<ptrdiff_t>(static_cast<size_t>(new_model.vertices_per_frame) - static_cast<size_t>(model.vertices_per_frame));
		const auto vertex_delta = static_cast<ptrdiff_t>(vertices.size() - model.vertices.size());
		const auto index_delta = static_cast<ptrdiff_t>(new_model.indices.size() - model.indices.size());

		const auto old_face_count = (model.indices.size() / model.face_vertex_count);
		const auto new_face_count = (new_model.indices.size() / new_model.face_vertex_count);
		const auto face_count_delta = static_cast<ptrdiff_t>(new_face_count - old_face_count);

		std::cout
			<< std::format("optimization stats:\n"
			               "\t verts per face: {} -> {}\n"
			               "\tverts per frame: {} -> {} ({})\n"
			               "\t    total verts: {} -> {} ({})\n"
			               "\t        indices: {} -> {} ({})\n"
			               "\t          faces: {} -> {} ({})\n",
			               static_cast<uint16_t>(model.face_vertex_count), static_cast<uint16_t>(new_model.face_vertex_count),
			               model.vertices_per_frame, new_model.vertices_per_frame, vert_frame_delta,
			               model.vertices.size(), vertices.size(), vertex_delta,
			               model.indices.size(), new_model.indices.size(), index_delta,
			               old_face_count, new_face_count, face_count_delta)
			<< std::endl;
	}

	if (options.stripify)
	{
		std::cout << "stripifying..." << std::endl;

		meshopt_optimizeVertexCacheStrip(new_model.indices.data(),
		                                 new_model.indices.data(),
		                                 new_model.indices.size(),
		                                 new_model.vertices_per_frame);

		const std::vector<uint16_t> strip_indices = stripify(new_model.indices, new_model.vertices_per_frame);
		size_t longest_strip = 0;

		new_model.strip_indices.reserve(strip_indices.size());

		for (auto full_strip_begin = strip_indices.begin();
		     full_strip_begin != strip_indices.end();)
		{
			const auto full_strip_end = std::find(full_strip_begin, strip_indices.end(), 0xFFFF);

			for (auto slice_begin = full_strip_begin; full_strip_begin != full_strip_end;)
			{
				const auto full_distance = std::distance(slice_begin, full_strip_end);

				if (full_distance == 3)
				{
					new_model.loose_tri_indices.insert(new_model.loose_tri_indices.end(), slice_begin, full_strip_end);
					break;
				}

				const auto slice_end = std::next(slice_begin, std::min<ptrdiff_t>(options.strip_max_points, full_distance));
				const auto slice_distance = std::distance(slice_begin, slice_end);

				if (slice_distance > 3)
				{
					longest_strip = std::max(longest_strip, static_cast<size_t>(slice_distance));
					new_model.strip_lengths.push_back(static_cast<uint16_t>(slice_distance));
					new_model.strip_indices.insert(new_model.strip_indices.end(), slice_begin, slice_end);
				}
				else
				{
					throw std::runtime_error("invalid strip length generated!");
				}

				if (full_distance == slice_distance)
				{
					break;
				}

				slice_begin = std::prev(slice_end, 2);
			}

			if (full_strip_end == strip_indices.end())
			{
				break;
			}

			full_strip_begin = std::next(full_strip_end);
		}

		new_model.strip_lengths.shrink_to_fit();
		new_model.strip_indices.shrink_to_fit();
		new_model.loose_tri_indices.shrink_to_fit();

		std::cout
			<< std::format("strip stats:\n"
			               "\tactual strips: {}\n"
			               "\t   loose tris: {}\n"
			               "\tlongest strip: {}\n"
			               "\tstrip indices: {}\n"
			               "\ttotal indices: {} (vs {})\n",
			               new_model.strip_lengths.size(),
			               new_model.loose_tri_indices.size() / 3,
			               longest_strip,
			               new_model.strip_indices.size(),
			               new_model.loose_tri_indices.size() + new_model.strip_indices.size(), new_model.indices.size())
			<< std::endl;

		if (new_model.strip_lengths.empty())
		{
			std::cout << "(failed; discarding changes)" << std::endl;
			stripify_failed = true;
		}
		else
		{
			new_model.flags |= RSDKModelFlags::is_stripped;
		}
	}

	// if simplification and stripification both failed, restore quad indices
	if (!quad_indices.empty() && new_model.face_vertex_count < model.face_vertex_count &&
	    simplify_failed && stripify_failed)
	{
		new_model.indices = std::move(quad_indices);
		new_model.face_vertex_count = model.face_vertex_count;
	}

	// now we convert back to RSDK format so we can save our changes
	new_model.vertices.resize(vertices.size());

	if (new_model.flags & RSDKModelFlags::use_textures)
	{
		new_model.tex_coords.resize(new_model.vertices_per_frame);
	}

	if (new_model.flags & RSDKModelFlags::use_colors)
	{
		new_model.colors.resize(new_model.vertices_per_frame);
	}

	for (size_t i = 0; i < vertices.size(); ++i)
	{
		const VertexForOptimizer& old_vert = vertices[i];
		RSDKModelVertex& new_vert = new_model.vertices[i];

		new_vert.position = old_vert.position;
		new_vert.normal = old_vert.normal;
	}

	if (new_model.flags & (RSDKModelFlags::use_textures | RSDKModelFlags::use_colors))
	{
		for (size_t i = 0; i < new_model.vertices_per_frame; ++i)
		{
			const VertexForOptimizer& old_vert = vertices[i];

			if (new_model.flags & RSDKModelFlags::use_textures)
			{
				RSDKTexCoord& tex_coord = new_model.tex_coords[i];
				tex_coord = old_vert.tex_coord;
			}

			if (new_model.flags & RSDKModelFlags::use_colors)
			{
				RSDKColor& new_color = new_model.colors[i];

				new_color.bytes[0] = static_cast<uint8_t>(old_vert.color.x * 255.0f);
				new_color.bytes[1] = static_cast<uint8_t>(old_vert.color.y * 255.0f);
				new_color.bytes[2] = static_cast<uint8_t>(old_vert.color.z * 255.0f);
				new_color.bytes[3] = static_cast<uint8_t>(old_vert.color.w * 255.0f);
			}
		}
	}

	if (options.bake_lighting)
	{
		std::cout << "baking lighting..." << std::endl;

		lighting_failed =
			!bake_lighting(new_model,
			               options.bake_light_direction,
			               options.bake_ambient_strength,
			               options.bake_diffuse_strength,
			               options.bake_diffuse_strength,
			               options.bake_specular_power);
	}

	if (optimize_failed &&
	    simplify_failed &&
	    stripify_failed &&
	    lighting_failed)
	{
		std::cout << "no changes were made; not writing new file." << std::endl;
		return 0;
	}

	if (options.output_file_path.has_parent_path())
	{
		const auto output_dir = options.output_file_path.parent_path();

		if (!std::filesystem::exists(output_dir))
		{
			if (options.create_output_dir)
			{
				std::cout << "creating output directory: " << output_dir << std::endl;

				if (!std::filesystem::create_directories(output_dir) &&
				    !std::filesystem::exists(output_dir))
				{
					std::cout << "failed! :( try creating it manually" << std::endl;
					return -1;
				}
			}
			else
			{
				std::cout << "output directory does not exist! please create it: " << output_dir << std::endl;
				return -1;
			}
		}
	}

	std::cout << "writing to file: " << options.output_file_path << std::endl;

	if (!write_model(options.output_file_path, new_model))
	{
		return -1;
	}

	std::cout << "done!" << std::endl;

	return 0;
}

bool equals_ignore_case(const std::string_view& a, const std::string_view& b)
{
	auto fn = [](unsigned char ca, unsigned char cb) { return std::tolower(ca) == std::tolower(cb); };
	return std::ranges::equal(a, b, fn);
}

bool is_arg(const std::string_view& str)
{
	return (str.length() == 2 && str[0] == '-' && !std::isdigit(str[1])) ||
	       str.starts_with("--");
}

bool parse_bool(const std::string_view& arg, bool* value)
{
	if (is_arg(arg))
	{
		return false;
	}

	if (arg == "1" ||
	    equals_ignore_case(arg, "true") ||
	    equals_ignore_case(arg, "on") ||
	    equals_ignore_case(arg, "yes"))
	{
		*value = true;
		return true;
	}

	if (arg == "0" ||
	    equals_ignore_case(arg, "false") ||
	    equals_ignore_case(arg, "off") ||
	    equals_ignore_case(arg, "no"))
	{
		*value = false;
		return true;
	}

	throw std::runtime_error(std::format("invalid value \"{}\" for argument", arg));
}

bool parse_float(const std::string_view& arg, float* value)
{
	if (is_arg(arg))
	{
		return false;
	}

	// note that the input format is dependent on the system locale which might not be desirable
	// (decimal place char)
	*value = std::stof(std::string(arg));
	return true;
}

bool parse_uint16(const std::string_view& arg, uint16_t* value)
{
	if (is_arg(arg))
	{
		return false;
	}

	*value = static_cast<uint16_t>(std::stoul(std::string(arg)));
	return true;
}

[[nodiscard]] std::optional<Options> parse_args(std::span<char*> args)
{
	Options options;

	for (size_t i = 1; i < args.size(); ++i)
	{
		const std::string_view arg(args[i]);

		try
		{
			if (arg == "-i" || arg == "--input")
			{
				if (i + 1 < args.size())
				{
					options.input_file_path = args[++i];
				}

				continue;
			}

			if (arg == "-o" || arg == "--output")
			{
				if (i + 1 < args.size())
				{
					options.output_file_path = args[++i];
				}

				continue;
			}

			if (arg == "--create-output-dir" || arg == "--create-output-directory")
			{
				options.create_output_dir = true;

				if (i + 1 < args.size() && parse_bool(args[i + 1], &options.create_output_dir))
				{
					++i;
				}

				continue;
			}

			if (arg == "--optimize")
			{
				options.optimize = true;

				if (i + 1 < args.size() && parse_bool(args[i + 1], &options.optimize))
				{
					++i;
				}

				continue;
			}

			if (arg == "--simplify")
			{
				options.simplify = true;

				if (i + 1 < args.size() && parse_bool(args[i + 1], &options.simplify))
				{
					++i;
				}

				continue;
			}

			if (arg == "--stripify")
			{
				options.stripify = true;

				if (i + 1 < args.size() && parse_bool(args[i + 1], &options.stripify))
				{
					++i;
				}

				continue;
			}

			if (arg == "--bake-lighting")
			{
				options.bake_lighting = true;

				if (i + 1 < args.size() && parse_bool(args[i + 1], &options.bake_lighting))
				{
					++i;
				}

				continue;
			}

			if (arg == "--strip-max-points")
			{
				if (++i == args.size() || !parse_uint16(args[i], &options.strip_max_points))
				{
					throw std::runtime_error("missing value for argument");
				}

				options.strip_max_points = std::max<uint16_t>(options.strip_max_points, 4);

				continue;
			}

			if (arg == "--simplify-index-threshold")
			{
				if (++i == args.size() || !parse_float(args[i], &options.simplify_index_threshold))
				{
					throw std::runtime_error("missing value for argument");
				}

				continue;
			}

			if (arg == "--simplify-target-error")
			{
				if (++i == args.size() || !parse_float(args[i], &options.simplify_target_error))
				{
					throw std::runtime_error("missing value for argument");
				}

				continue;
			}

			if (arg == "--bake-light-direction")
			{
				if (args.size() - i < 4 ||
				    !parse_float(args[++i], &options.bake_light_direction.x) ||
				    !parse_float(args[++i], &options.bake_light_direction.y) ||
				    !parse_float(args[++i], &options.bake_light_direction.z))
				{
					throw std::runtime_error("missing values for argument");
				}

				continue;
			}

			if (arg == "--bake-ambient-strength")
			{
				if (++i == args.size() || !parse_float(args[i], &options.bake_ambient_strength))
				{
					throw std::runtime_error("missing value for argument");
				}

				continue;
			}

			if (arg == "--bake-diffuse-strength")
			{
				if (++i == args.size() || !parse_float(args[i], &options.bake_diffuse_strength))
				{
					throw std::runtime_error("missing value for argument");
				}

				continue;
			}

			if (arg == "--bake-specular-strength")
			{
				if (++i == args.size() || !parse_float(args[i], &options.bake_specular_strength))
				{
					throw std::runtime_error("missing value for argument");
				}

				continue;
			}

			if (arg == "--bake-specular-power")
			{
				if (++i == args.size() || !parse_float(args[i], &options.bake_specular_power))
				{
					throw std::runtime_error("missing value for argument");
				}

				continue;
			}
		}
		catch (const std::exception& ex)
		{
			std::cerr << "error parsing argument \"" << arg << "\": " << ex.what() << std::endl;
			return std::nullopt;
		}
	}

	return options;
}
