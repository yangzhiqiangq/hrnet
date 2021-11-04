
#include "hrnet.h"
#include <iomanip>
TransitiongrasonImpl::TransitiongrasonImpl(int cur_index_,
										   std::vector<int> num_channels_pre, std::vector<int> num_channels_cur)
{
	cur_index = cur_index_;
	n_pre = num_channels_pre.size();
	n_cur = num_channels_cur.size();
	for (int j = 0; j < cur_index + 1 - n_pre; j++)
	{
		int in_channel = num_channels_pre.back();
		int out_channel;
		if (j == cur_index - n_pre)
		{
			out_channel = num_channels_cur[cur_index];
		}
		else
		{
			out_channel = in_channel;
		}
		torch::nn::Sequential se;
		se->push_back(torch::nn::Conv2d(conv_options(in_channel, out_channel, 3, 2, 1, false)));
		se->push_back(torch::nn::BatchNorm2d(out_channel));
		se->push_back(torch::nn::ReLU());
		se = register_module(std::to_string(j), se);
		seqs.push_back(se);
	}
}

torch::Tensor TransitiongrasonImpl::forward(std::vector<torch::Tensor> x)
{
	torch::Tensor re;
	for (int j = 0; j < cur_index + 1 - n_pre; j++)
	{
		if (j == 0)
		{
			re = seqs[j]->forward(x[n_pre - 1]);
		}
		else
		{
			re += (seqs[j]->forward(x[n_pre - 1]));
		}
	}
	return re;
}
TransitionsonImpl::TransitionsonImpl(std::vector<int> num_channels_pre,
									 std::vector<int> num_channels_cur)
{
	n_pre = num_channels_pre.size();
	n_cur = num_channels_cur.size();
	num_channels_pre_ = num_channels_pre;
	num_channels_cur_ = num_channels_cur;
	for (int i = 0; i < n_cur; i++)
	{
		if (i < n_pre)
		{
			if (num_channels_pre[i] != num_channels_cur[i])
			{
				torch::nn::Sequential se;
				//torch::nn::Conv2d con = torch::nn::Conv2d(conv_options(num_channels_pre[i], num_channels_cur[i], 3, 1, 0, false));
				//torch::nn::BatchNorm2d bn = torch::nn::BatchNorm2d(num_channels_cur[i]);
				se->push_back(torch::nn::Conv2d(conv_options(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, false)));
				se->push_back(torch::nn::BatchNorm2d(num_channels_cur[i]));
				se->push_back(torch::nn::ReLU());
				se = register_module(std::to_string(i), se);
				seqs.push_back(se);
			}
			else
			{
				seqs.push_back(nullptr);
			}
		}
		else
		{
			Transitiongrason son = Transitiongrason(i, num_channels_pre, num_channels_cur);
			son = register_module(std::to_string(i), son);
			grasonlist.push_back(son);
		}
	}
}

std::vector<torch::Tensor> TransitionsonImpl::forward(std::vector<torch::Tensor> x)
{
	std::vector<torch::Tensor> re(n_cur);
	for (int i = 0; i < n_cur; i++)
	{

		if (i < n_pre)
		{
			if (seqs[i])
			{
				re[i] = seqs[i]->forward(x[i]);
			}
			else
			{
				re[i] = x[i];
			}
		}
		else
		{
			re[i] = grasonlist[i - n_pre]->forward(x);
		}
	}
	return re;
}

HrNetImpl::HrNetImpl(int in_channel, std::vector<int> num_modules, std::vector<int> num_branches, std::vector<std::vector<int>> num_blocks, std::vector<std::vector<int>> num_channels)
{
	conv1 = torch::nn::Conv2d(conv_options(in_channel, 64, 3, 2, 1, false));
	torch::nn::BatchNorm2dOptions bnn = torch::nn::BatchNorm2dOptions(64);
	bn1 = torch::nn::BatchNorm2d(bnn);
	conv2 = torch::nn::Conv2d(conv_options(64, 64, 3, 2, 1, false));
	torch::nn::BatchNorm2dOptions bnn1 = torch::nn::BatchNorm2dOptions(64);
	bn2 = torch::nn::BatchNorm2d(bnn1);
	conv1 = register_module("conv1", conv1);
	bn1 = register_module("bn1", bn1);
	conv2 = register_module("conv2", conv2);
	bn2 = register_module("bn2", bn2);

	layer1 = Layer1(64, 64, 4);
	layer1 = register_module("layer1", layer1);
	num_channels[0][0] *= 4;
	transition1 = Transitionson(num_channels[0], num_channels[1]);
	transition2 = Transitionson(num_channels[1], num_channels[2]);
	transition3 = Transitionson(num_channels[2], num_channels[3]);
	transition1 = register_module("transition1", transition1);
	transition2 = register_module("transition2", transition2);
	transition3 = register_module("transition3", transition3);

	stage2 = StageNum(num_modules[1], num_branches[1], num_blocks[1], num_channels[1]);
	stage3 = StageNum(num_modules[2], num_branches[2], num_blocks[2], num_channels[2]);
	stage4 = StageNum(num_modules[3], num_branches[3], num_blocks[3], num_channels[3]);
	stage2 = register_module("stage2", stage2);
	stage3 = register_module("stage3", stage3);
	stage4 = register_module("stage4", stage4);
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

std::vector<torch::Tensor> HrNetImpl::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = bn1->forward(x);
	x = torch::relu(x);
	x = conv2->forward(x);
	x = bn2->forward(x);
	x = torch::relu(x);
	x = layer1->forward(x);

	std::vector<torch::Tensor> xlist;
	xlist.push_back(x);
	xlist = transition1->forward(xlist);
	xlist = stage2->forward(xlist);
	xlist = transition2->forward(xlist);
	xlist = stage3->forward(xlist);
	xlist = transition3->forward(xlist);
	xlist = stage4->forward(xlist);
	return xlist;
}
