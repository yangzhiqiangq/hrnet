#include "hrmodule.h"
//stage中输出的branch大于输入的branch时的情况
class TransitiongrasonImpl : public torch::nn::Module
{
public:
	TransitiongrasonImpl(int cur_index_,
						 std::vector<int> num_channels_pre, std::vector<int> num_channels_cur);
	torch::Tensor forward(std::vector<torch::Tensor>);

private:
	int cur_index;
	int n_cur, n_pre;
	std::vector<torch::nn::Sequential> seqs;
};
TORCH_MODULE(Transitiongrason);

//不同stage进行转换
class TransitionsonImpl : public torch::nn::Module
{
public:
	TransitionsonImpl(std::vector<int> num_channels_pre,
					  std::vector<int> num_channels_cur);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor>);

private:
	int n_pre, n_cur;
	std::vector<int> num_channels_pre_;
	std::vector<int> num_channels_cur_;
	std::vector<torch::nn::Sequential> seqs;
	std::vector<Transitiongrason> grasonlist;
};
TORCH_MODULE(Transitionson);

//Hrnet网络
class HrNetImpl : public torch::nn::Module
{
public:
	HrNetImpl(int in_channel, std::vector<int> num_modules, std::vector<int> num_branches,
			  std::vector<std::vector<int>> num_blocks, std::vector<std::vector<int>> num_channels);
	std::vector<torch::Tensor> forward(torch::Tensor);

private:
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Conv2d conv2{nullptr};
	torch::nn::BatchNorm2d bn2{nullptr};
	Layer1 layer1 = nullptr;
	Transitionson transition1 = nullptr;
	Transitionson transition2 = nullptr;
	Transitionson transition3 = nullptr;

	StageNum stage2 = nullptr;
	StageNum stage3 = nullptr;
	StageNum stage4 = nullptr;
};
TORCH_MODULE(HrNet);