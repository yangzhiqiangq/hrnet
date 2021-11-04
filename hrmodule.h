#include "fuse.h"
//残差网络
class BlockhrImpl : public torch::nn::Module
{
public:
	BlockhrImpl(int channel);
	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Conv2d con1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Conv2d con2{nullptr};
	torch::nn::BatchNorm2d bn2{nullptr};
};
TORCH_MODULE(Blockhr);

//多个branchs和blocks 组成网络
class HrBranchImpl : public torch::nn::Module
{
public:
	HrBranchImpl(int num_branches_, std::vector<int> num_blocks_, std::vector<int> num_channels);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);

private:
	std::vector<torch::nn::Sequential> seqs;
	int num_branches;
	std::vector<int> num_blocks;
};
TORCH_MODULE(HrBranch);

//hrbranchlayer 和 fuselayer组成
class HrModuleImpl : public torch::nn::Module
{
public:
	HrModuleImpl(int ind, int num_branches_, std::vector<int> num_blocks_, std::vector<int> num_channels);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);

private:
	std::vector<torch::nn::Sequential> seqs;
	int num_branches;
	int ind;
	std::vector<int> num_blocks;
	ThreeModule fuse = nullptr;
	HrBranch branch = nullptr;
};
TORCH_MODULE(HrModule);

// stage网络 每个stage有n个hrModule
class StageNumImpl : public torch::nn::Module
{
public:
	StageNumImpl(int num_modules_, int num_branches_, std::vector<int> num_blocks_,
				 std::vector<int> num_channels);
	std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);

private:
	int num_modules;
	std::vector<HrModule> hrmodulelist;
};
TORCH_MODULE(StageNum);

class DownSampleImpl : public torch::nn::Module
{
public:
	DownSampleImpl(int channel);
	torch::Tensor forward(torch::Tensor);

private:
	torch::nn::Sequential seqs;
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	int expaision = 4;
};
TORCH_MODULE(DownSample);

//stage1中的残差网络
class BOTTLENECKImpl : public torch::nn::Module
{
public:
	BOTTLENECKImpl(int index, int in_channel, int channel);
	torch::Tensor forward(torch::Tensor x);

private:
	int expaision = 4;
	int in;
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Conv2d conv2{nullptr};
	torch::nn::BatchNorm2d bn2{nullptr};
	torch::nn::Conv2d conv3{nullptr};
	torch::nn::BatchNorm2d bn3{nullptr};
	//DownSample downsample = nullptr;
	torch::nn::Sequential seqs;
};
TORCH_MODULE(BOTTLENECK);
//wangluo 中的stage1
class Layer1Impl : public torch::nn::Module
{
public:
	Layer1Impl(int inplanes, int planes, int blocks);
	torch::Tensor forward(torch::Tensor x);

private:
	std::vector<BOTTLENECK> blocklist;
};
TORCH_MODULE(Layer1);
