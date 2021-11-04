
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <fstream>

inline torch::Tensor resizeup(torch::Tensor x, std::vector<int64_t> si)
{
	namespace F = torch::nn::functional;
	F::InterpolateFuncOptions optional = F::InterpolateFuncOptions();
	optional.size(si);
	optional.mode(torch::kBilinear);
	optional.align_corners(false);
	return F::interpolate(x, optional);
}

//fuselayer时 输入层的通道小于输出层的通道的情况
class OneModuleImpl : public torch::nn::Module
{
public:
	OneModuleImpl(int i, int j, std::vector<int> in_channels);
	torch::Tensor forward(torch::Tensor);

private:
	torch::nn::Sequential conv1;
	torch::nn::Sequential conv2;
	torch::nn::Sequential conv3;
	torch::nn::Sequential conv4;
};
TORCH_MODULE(OneModule);

//fuselayer中 单一输入层对应的多个输出
class TwoModuleImpl : public torch::nn::Module
{
public:
	TwoModuleImpl(int i, int nj, std::vector<int> in_channels);
	torch::Tensor forward(std::vector<torch::Tensor> x);

private:
	int njj;
	int index;
	torch::nn::Sequential conv1;
	torch::nn::Sequential conv2;
	torch::nn::Sequential conv3;
	OneModule convd1 = nullptr;
	OneModule convd2 = nullptr;
	OneModule convd3 = nullptr;
};
TORCH_MODULE(TwoModule);

//fuselayer  多个输入对应多个输出
class ThreeModuleImpl : public torch::nn::Module
{
public:
	ThreeModuleImpl(int ni, std::vector<int> in_channels);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);

private:
	int n;
	std::vector<torch::Tensor> re;
	TwoModule tm1 = nullptr;
	TwoModule tm2 = nullptr;
	TwoModule tm3 = nullptr;
	TwoModule tm4 = nullptr;
};
TORCH_MODULE(ThreeModule);

class upsamImpl : public torch::nn::Module
{
public:
	upsamImpl(std::vector<double> scale_factor_, bool align_corners_);
	torch::Tensor forward(torch::Tensor);

private:
	std::vector<double> scale_factor;
	bool align_corners;
};
TORCH_MODULE(upsam);
inline torch::nn::Conv2dOptions conv_options(int in_planes, int out_planes, int kerner_size,
											 int stride = 1, int padding = 0, bool with_bias = true)
{
	torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
	conv_options.stride(stride);
	conv_options.padding(padding);
	conv_options.bias(with_bias);
	return conv_options;
}

inline torch::nn::UpsampleOptions upsample_options(std::vector<double> scale_size, bool align_corners = true)
{
	torch::nn::UpsampleOptions upsample_options = torch::nn::UpsampleOptions();
	upsample_options.scale_factor(scale_size);
	upsample_options.mode(torch::kBilinear).align_corners(align_corners);
	return upsample_options;
}

//void xietxt(torch::Tensor x, std::string s)
//{
//	std::ofstream dout(s);
//	for (int i = 0; i < x.size(1); i++)
//	{
//		for (int j = 0; j < x.size(2); j++)
//		{
//			for (int k = 0; k < x.size(3); k++)
//			{
//				dout << (x[0][i][j][k]) << std::endl;
//			}
//		}
//	}
//
//}