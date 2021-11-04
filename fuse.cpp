
#include "fuse.h"
//Hrnet网络中fuselayer中输出branch的索引大于输入branch索引情况，i>j
//i为输出索引，j为输出索引，in_channels为 每一个branch中channel的数组
OneModuleImpl::OneModuleImpl(int i, int j, std::vector<int> in_channels)
{

	for (int k = 0; k < i - j; k++)
	{
		torch::nn::Conv2d con{nullptr};
		torch::nn::BatchNorm2d bn{nullptr};
		if (k == 0)
		{
			if (k == i - j - 1)
			{
				con = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[i], 3, 2, 1, false));
				bn = torch::nn::BatchNorm2d(in_channels[i]);
				conv1->push_back(con);
				conv1->push_back(bn);
			}
			else
			{
				con = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[j], 3, 2, 1, false));
				bn = torch::nn::BatchNorm2d(in_channels[j]);
				conv1->push_back(con);
				conv1->push_back(bn);
				conv1->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(false)));
			}
			conv1 = register_module("0", conv1);
		}
		if (k == 1)
		{
			if (k == i - j - 1)
			{
				con = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[i], 3, 2, 1, false));
				bn = torch::nn::BatchNorm2d(in_channels[i]);
				conv2->push_back(con);
				conv2->push_back(bn);
			}
			else
			{
				con = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[j], 3, 2, 1, false));
				bn = torch::nn::BatchNorm2d(in_channels[j]);
				conv2->push_back(con);
				conv2->push_back(bn);
				conv2->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(false)));
			}
			conv2 = register_module("1", conv2);
		}
		if (k == 2)
		{
			if (k == i - j - 1)
			{
				con = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[i], 3, 2, 1, false));
				bn = torch::nn::BatchNorm2d(in_channels[i]);
				conv3->push_back(con);
				conv3->push_back(bn);
			}
			else
			{
				con = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[j], 3, 2, 1, false));
				bn = torch::nn::BatchNorm2d(in_channels[j]);
				conv3->push_back(con);
				conv3->push_back(bn);
				conv3->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(false)));
			}
			conv3 = register_module("2", conv3);
		}
		if (k == 3)
		{
			if (k == i - j - 1)
			{
				con = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[i], 3, 2, 1, false));
				bn = torch::nn::BatchNorm2d(in_channels[i]);
				conv4->push_back(con);
				conv4->push_back(bn);
			}
			else
			{
				con = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[j], 3, 2, 1, false));
				bn = torch::nn::BatchNorm2d(in_channels[j]);
				conv4->push_back(con);
				conv4->push_back(bn);
				conv4->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(false)));
			}
			conv4 = register_module("3", conv4);
		}
	}
}
torch::Tensor OneModuleImpl::forward(torch::Tensor x)
{

	x = conv1->forward(x);
	if (!conv2->is_empty())
		x = conv2->forward(x);
	if (!conv3->is_empty())
		x = conv3->forward(x);
	if (!conv4->is_empty())
		x = conv4->forward(x);

	return x;
}

//hrnet网络中fuselayer中输出的单一branch对应多个输入branch
//i为输出branch的索引，nj为输入branch的数量，in_channels 多个branch中channel的数组
TwoModuleImpl::TwoModuleImpl(int i, int nj, std::vector<int> in_channels)
{

	njj = nj;
	index = i;
	int scale = 1;
	for (int j = 0; j < nj; j++)
	{
		scale = 1;
		for (int k = i; k < j; k++)
			scale *= 2;
		torch::nn::Conv2d con1{nullptr};
		upsam con2 = nullptr;
		torch::nn::BatchNorm2d bn{nullptr};
		std::vector<double> scales = {double(scale), double(scale)};
		if (j > i)
		{
			if (j == 1)
			{
				con1 = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[i], 1, 1, 0, false));
				bn = torch::nn::BatchNorm2d(in_channels[i]);
				con2 = upsam(scales, false);
				conv1->push_back(con1);
				conv1->push_back(bn);
				conv1->push_back(con2);
				conv1 = register_module("1", conv1);
			}
			if (j == 2)
			{
				con1 = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[i], 1, 1, 0, false));
				bn = torch::nn::BatchNorm2d(in_channels[i]);
				con2 = upsam(scales, false);
				conv2->push_back(con1);
				conv2->push_back(bn);
				conv2->push_back(con2);
				conv2 = register_module("2", conv2);
			}
			if (j == 3)
			{
				con1 = torch::nn::Conv2d(conv_options(in_channels[j], in_channels[i], 1, 1, 0, false));
				bn = torch::nn::BatchNorm2d(in_channels[i]);
				con2 = upsam(scales, false);
				conv3->push_back(con1);
				conv3->push_back(bn);
				conv3->push_back(con2);
				conv3 = register_module("3", conv3);
			}
		}
		else if (i > j)
		{
			if (j == 0)
			{
				convd1 = OneModule(i, j, in_channels);
				convd1 = register_module("0", convd1);
			}
			if (j == 1)
			{
				convd2 = OneModule(i, j, in_channels);
				convd2 = register_module("1", convd2);
			}
			if (j == 2)
			{
				convd3 = OneModule(i, j, in_channels);
				convd3 = register_module("2", convd3);
			}
		}
	}
}

torch::Tensor TwoModuleImpl::forward(std::vector<torch::Tensor> x)
{
	torch::Tensor re = torch::zeros_like(x[index]);
	for (int i = 0; i < njj; i++)
	{
		if (i == 0)
		{
			if (index == i)
			{
				re += x[i];
			}
			else if (index > i)
			{
				re += (convd1->forward(x[i]));
			}
		}
		if (i == 1)
		{
			if (index == i)
			{
				re += x[i];
			}
			else if (index > i)
			{
				re += (convd2->forward(x[i]));
			}
			else
			{
				re += resizeup(conv1->forward(x[i]), std::vector<int64_t>{x[index].size(2), x[index].size(3)});
			}
		}
		if (i == 2)
		{
			if (index == i)
			{
				re += x[i];
			}
			else if (index > i)
			{

				re += (convd3->forward(x[i]));
			}
			else
			{
				re += resizeup(conv2->forward(x[i]), std::vector<int64_t>{x[index].size(2), x[index].size(3)});
			}
		}
		if (i == 3)
		{
			if (index == i)
			{
				re += x[i];
			}
			else if (index < i)
			{
				re += resizeup(conv3->forward(x[i]), std::vector<int64_t>{x[index].size(2), x[index].size(3)});
			}
		}
	}
	re = torch::relu(re);
	return re;
}

//hrnet中fuselayer的类 
//ni为要融合branch的数量，in_channels为各个branch中通道数量
ThreeModuleImpl::ThreeModuleImpl(int ni, std::vector<int> in_channels)
{
	n = ni;
	for (int i = 0; i < ni; i++)
	{
		if (i == 0)
		{
			tm1 = TwoModule(i, n, in_channels);
			tm1 = register_module("0", tm1);
		}
		if (i == 1)
		{
			tm2 = TwoModule(i, n, in_channels);
			tm2 = register_module("1", tm2);
		}
		if (i == 2)
		{
			tm3 = TwoModule(i, n, in_channels);
			tm3 = register_module("2", tm3);
		}
		if (i == 3)
		{
			tm4 = TwoModule(i, n, in_channels);
			tm4 = register_module("3", tm4);
		}
	}
}
std::vector<torch::Tensor> ThreeModuleImpl::forward(std::vector<torch::Tensor> x)
{
	for (int i = 0; i < n; i++)
	{
		if (i == 0)
		{
			re.push_back(tm1->forward(x));
		}
		if (i == 1)
		{
			re.push_back(tm2->forward(x));
		}
		if (i == 2)
		{
			re.push_back(tm3->forward(x));
		}
		if (i == 3)
		{
			re.push_back(tm4->forward(x));
		}
	}

	return re;
}
//上采样类 
//scale_factor 为上采样的比例 如：{2.0，2.0} 扩大2倍
upsamImpl::upsamImpl(std::vector<double> scale_factor_, bool align_corners_ = false)
{
	scale_factor = scale_factor_;
	align_corners = align_corners_;
}
torch::Tensor upsamImpl::forward(torch::Tensor x)
{
	namespace F = torch::nn::functional;
	F::InterpolateFuncOptions optional = F::InterpolateFuncOptions();
	optional.scale_factor(scale_factor);
	optional.mode(torch::kBilinear);
	optional.align_corners(false);
	return F::interpolate(x, optional);
}