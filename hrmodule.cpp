#include "hrmodule.h"
#include <string>
BlockhrImpl::BlockhrImpl(int channel)
{
	con1 = torch::nn::Conv2d(conv_options(channel, channel, 3, 1, 1, false));
	bn1 = torch::nn::BatchNorm2d(channel);
	con2 = torch::nn::Conv2d(conv_options(channel, channel, 3, 1, 1, false));
	bn2 = torch::nn::BatchNorm2d(channel);
	con1 = register_module("conv1", con1);
	bn1 = register_module("bn1", bn1);
	con2 = register_module("conv2", con2);
	bn2 = register_module("bn2", bn2);
}
torch::Tensor BlockhrImpl::forward(torch::Tensor x)
{
	torch::Tensor residual = x.clone();
	x = con1->forward(x);
	x = bn1->forward(x);
	x = torch::relu(x);
	x = con2->forward(x);
	x = bn2->forward(x);
	x += residual;
	x = torch::relu(x);
	return x;
}

HrBranchImpl::HrBranchImpl(int num_branches_, std::vector<int> num_blocks_, std::vector<int> num_channels_)
{
	num_branches = num_branches_;
	num_blocks = num_blocks_;
	for (int i = 0; i < num_branches; i++)
	{
		torch::nn::Sequential se;
		for (int j = 0; j < num_blocks[i]; j++)
		{
			Blockhr b(num_channels_[i]);
			se->push_back(b);
		}
		se = register_module(std::to_string(i), se);
		seqs.push_back(se);
	}
}
std::vector<torch::Tensor> HrBranchImpl::forward(std::vector<torch::Tensor> x)
{
	for (int i = 0; i < num_branches; i++)
	{
		x[i] = seqs[i]->forward(x[i]);
	}
	return x;
}

HrModuleImpl::HrModuleImpl(int ind_, int num_branches_, std::vector<int> num_blocks_, std::vector<int> num_channels_)
{
	num_branches = num_branches_;
	num_blocks = num_blocks_;
	ind = ind_;
	branch = HrBranch(num_branches, num_blocks, num_channels_);
	branch = register_module("branches", branch);
	fuse = ThreeModule(num_branches, num_channels_);
	fuse = register_module("fuse_layers", fuse);
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
//				dout << (x[0][i][j][k].item<float>()) << std::endl;
//			}
//		}
//	}
//
//}

std::vector<torch::Tensor> HrModuleImpl::forward(std::vector<torch::Tensor> x)
{
	x = branch->forward(x);
	x = fuse->forward(x);
	return x;
}

StageNumImpl::StageNumImpl(int num_modules_, int num_branches_, std::vector<int> num_blocks_, std::vector<int> num_channels)
{
	num_modules = num_modules_;
	for (int i = 0; i < num_modules_; i++)
	{
		HrModule hm = HrModule(i, num_branches_, num_blocks_, num_channels);
		hm = register_module(std::to_string(i), hm);
		hrmodulelist.push_back(hm);
	}
}

std::vector<torch::Tensor> StageNumImpl::forward(std::vector<torch::Tensor> x)
{
	for (int i = 0; i < num_modules; i++)
	{
		x = hrmodulelist[i]->forward(x);
	}
	return x;
}

DownSampleImpl::DownSampleImpl(int channel)
{
	seqs->push_back(torch::nn::Conv2d(conv_options(channel, expaision * channel, 1, 1, 0, false)));
	seqs->push_back(torch::nn::BatchNorm2d(expaision * channel));
}
torch::Tensor DownSampleImpl::forward(torch::Tensor x)
{
	return seqs->forward(x);
}

BOTTLENECKImpl::BOTTLENECKImpl(int index, int in_channel, int channel)
{
	in = index;
	if (index > 0)
	{
		in_channel = channel * 4;
	}
	conv1 = torch::nn::Conv2d(conv_options(in_channel, channel, 1, 1, 0, false));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channel));

	conv2 = torch::nn::Conv2d(conv_options(channel, channel, 3, 1, 1, false));
	bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channel));

	conv3 = torch::nn::Conv2d(conv_options(channel, channel * 4, 1, 1, 0, false));
	bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channel * 4));

	conv1 = register_module("conv1", conv1);
	conv2 = register_module("conv2", conv2);
	conv3 = register_module("conv3", conv3);
	bn1 = register_module("bn1", bn1);
	bn2 = register_module("bn2", bn2);
	bn3 = register_module("bn3", bn3);

	if (index == 0)
	{
		seqs->push_back(torch::nn::Conv2d(conv_options(channel, expaision * channel, 1, 1, 0, false)));
		seqs->push_back(torch::nn::BatchNorm2d(expaision * channel));
		seqs = register_module("downsample", seqs);
	}
}

torch::Tensor BOTTLENECKImpl::forward(torch::Tensor x)
{
	torch::Tensor out;
	torch::Tensor residual = x.clone();
	int row = 94;
	int col = 94;
	out = conv1->forward(x);
	out = bn1->forward(out);
	out = torch::relu(out);
	out = conv2->forward(out);
	out = bn2->forward(out);
	out = torch::relu(out);
	out = conv3->forward(out);

	out = bn3->forward(out);
	if (in == 0)
	{
		residual = seqs->forward(residual);
		out += residual;
	}
	else
		out += residual;
	out = torch::relu(out);
	return out;
}
//henet中stage1
Layer1Impl::Layer1Impl(int inplanes, int planes, int blocks)
{
	for (int i = 0; i < blocks; i++)
	{
		BOTTLENECK bo = BOTTLENECK(i, inplanes, planes);
		bo = register_module(std::to_string(i), bo);
		blocklist.push_back(bo);
	}
}

torch::Tensor Layer1Impl::forward(torch::Tensor x)
{
	for (int i = 0; i < blocklist.size(); i++)
	{
		x = blocklist[i]->forward(x);
	}
	return x;
}
