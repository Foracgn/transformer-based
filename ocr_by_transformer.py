# -*- coding: utf-8 -*-
"""

use transformer to do OCR!

利用transformer来完成一个简单的OCR字符识别任务

@author: anshengmath@163.com
"""
import os
from PIL import Image

import torchvision.models as models
import torchvision.transforms as transforms

import logging

from analysis_recognition_dataset import load_lbl2id_map, statistics_max_len_label
from transformer import *
from train_utils import *

logging.basicConfig(filename="logs",
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class RecognitionDataset(object):

    def __init__(self, dataset_root_dir, lbl2id_map, sequence_len, max_ratio, phase='train', pad=0):

        if phase == 'train':
            self.img_dir = os.path.join(base_data_dir, 'train')
            self.lbl_path = os.path.join(base_data_dir, 'train_gt.txt')
        else:
            self.img_dir = os.path.join(base_data_dir, 'valid')
            self.lbl_path = os.path.join(base_data_dir, 'valid_gt.txt')
        self.lbl2id_map = lbl2id_map
        self.pad = pad  # padding标识符的id，默认0
        self.sequence_len = sequence_len  # 序列长度
        self.max_ratio = max_ratio * 3  # 将宽拉长3倍

        self.imgs_list = []
        self.lbls_list = []
        with open(self.lbl_path, 'r') as reader:
            for line in reader:
                items = line.rstrip().split(',')
                img_name = items[0]
                lbl_str = items[1].strip()[1:-1]

                self.imgs_list.append(img_name)
                self.lbls_list.append(lbl_str)

        # 定义随机颜色变换
        self.color_trans = transforms.ColorJitter(0.1, 0.1, 0.1)
        # 定义 Normalize
        self.trans_Normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        """ 
        获取对应index的图像和ground truth label，并视情况进行数据增强
        """
        imgName = self.imgs_list[index]
        imgPath = os.path.join(self.img_dir, imgName)
        lbl_str = self.lbls_list[index]

        # ----------------
        # 图片预处理
        # ----------------
        # load image
        img = Image.open(imgPath).convert('RGB')

        # 对图片进行大致等比例的缩放
        # 将高缩放到32，宽大致等比例缩放，但要被32整除
        w, h = img.size
        ratio = round((w / h) * 3)  # 将宽拉长3倍，然后四舍五入
        if ratio == 0:
            ratio = 1
        if ratio > self.max_ratio:
            ratio = self.max_ratio
        h_new = 32
        w_new = h_new * ratio
        img_resize = img.resize((w_new, h_new), Image.BILINEAR)

        # 对图片右半边进行padding，使得宽/高比例固定=self.max_ratio
        imgPadding = Image.new('RGB', (32 * self.max_ratio, 32), (0, 0, 0))
        imgPadding.paste(img_resize, (0, 0))

        # 随机颜色变换
        imgInput = self.color_trans(imgPadding)
        # Normalize
        imgInput = self.trans_Normalize(imgInput)

        # ----------------
        # label处理
        # ----------------

        # 构造encoder的mask
        encodeMask = [1] * ratio + [0] * (self.max_ratio - ratio)
        encodeMask = torch.tensor(encodeMask)
        encodeMask = (encodeMask != 0).unsqueeze(0)

        # 构造ground truth label
        gt = [1]
        for lbl in lbl_str:
            gt.append(self.lbl2id_map[lbl])
        gt.append(2)
        for i in range(len(lbl_str), self.sequence_len):  # 除去起始符终止符，lbl长度为sequence_len，剩下的padding
            gt.append(0)
        # 截断为预设的最大序列长度
        gt = gt[:self.sequence_len + 2]

        # decoder的输入
        decodeIn = gt[:-1]
        decodeIn = torch.tensor(decodeIn)
        # decoder的输出
        decodeOut = gt[1:]
        decodeOut = torch.tensor(decodeOut)
        # decoder的mask 
        decodeMask = self.make_std_mask(decodeIn, self.pad)
        # 有效tokens数
        nTokens = (decodeOut != self.pad).data.sum()

        return imgInput, encodeMask, decodeIn, decodeOut, decodeMask, nTokens

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        padd 和 future words 均在mask中用0表示
        """
        tgt_mask = (tgt != pad)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        tgt_mask = tgt_mask.squeeze(0)  # subsequent返回值的shape是(1, N, N)
        return tgt_mask

    def __len__(self):
        return len(self.imgs_list)


# Model Architecture
class OCR_EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. 
    Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, src_position, tgt_embed, generator):
        super(OCR_EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # input embedding module(input embedding + positional encode)
        self.src_position = src_position
        self.tgt_embed = tgt_embed  # output embedding module
        self.generator = generator  # output generation module

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        memory = self.encode(src, src_mask)
        res = self.decode(memory, src_mask, tgt, tgt_mask)
        return res

    def encode(self, src, src_mask):
        # feature extract
        src_embedds = self.src_embed(src)
        # 将src_embedds由shape(bs, model_dim, 1, max_ratio) 处理为transformer期望的输入shape(bs, 时间步, model_dim)
        src_embedds = src_embedds.squeeze(-2)
        src_embedds = src_embedds.permute(0, 2, 1)

        # position encode
        src_embedds = self.src_position(src_embedds)

        return self.encoder(src_embedds, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        target_embedds = self.tgt_embed(tgt)
        return self.decoder(target_embedds, memory, src_mask, tgt_mask)


def make_ocr_model(tgt_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建模型
    params:
        tgt_vocab: 输出的词典大小(82)
        N: 编码器和解码器堆叠基础模块的个数
        d_model: 模型中embedding的size，默认512
        d_ff: FeedForward Layer层中embedding的size，默认2048
        h: MultiHeadAttention中多头的个数，必须被d_model整除
        dropout:
    """
    c = copy.deepcopy

    backbone = models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])  # 去掉最后两个层 (global average pooling and fc layer)

    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = OCR_EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n),
        backbone,
        c(position),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # Initialize parameters with Gloot / fan_avg.
    for child in model.children():
        if child is backbone:
            # 将backbone的权重设为不计算梯度
            for param in child.parameters():
                param.requires_grad = False
            # 预训练好的backbone不进行随机初始化，其余模块进行随机初始化
            continue
        for p in child.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model


def run_epoch(data_loader, model, loss_compute, device=None):
    """Standard Training and Logging Function"""
    start = time.time()
    totalTokens = 0
    totalLoss = 0
    tokens = 0
    for i, batch in enumerate(data_loader):
        # if device == "cuda":
        #    batch.to_device(device) 
        imgInput, encodeMask, decodeIn, decodeOut, decodeMask, nTokens = batch
        imgInput = imgInput.to(device)
        encodeMask = encodeMask.to(device)
        decodeIn = decodeIn.to(device)
        decodeOut = decodeOut.to(device)
        decodeMask = decodeMask.to(device)
        nTokens = torch.sum(nTokens).to(device)

        out = model.forward(imgInput, decodeIn, encodeMask, decodeMask)

        loss = loss_compute(out, decodeOut, nTokens)
        totalLoss += loss
        totalTokens += nTokens
        tokens += nTokens
        if i % 50 == 1:
            elapsed = time.time() - start
            logger.info("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                        (i, loss / nTokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return totalLoss / totalTokens


# greedy decode
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    memory = model.encode(src, src_mask)
    # ys代表目前已生成的序列，最初为仅包含一个起始符的序列，不断将预测结果追加到序列最后
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long()
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        next_word = torch.ones(1, 1).type_as(src.data).fill_(next_word).long()
        ys = torch.cat([ys, next_word], dim=1)

        next_word = int(next_word)
        if next_word == end_symbol:
            break
        # ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    ys = ys[0, 1:]
    return ys


def judge_is_correct(prediction, label):
    # 判断模型预测结果和label是否一致
    predictionLen = prediction.shape[0]
    label = label[:predictionLen]
    isCorrect = 1 if label.equal(prediction) else 0
    return isCorrect


if __name__ == "__main__":

    # TODO set parameters
    base_data_dir = './ICDAR_2015/'  # 数据集根目录，请将数据下载到此位置
    device = torch.device('cuda')
    nRofEpochs = 1000
    batch_size = 64
    model_save_path = './log/ex1_ocr_model.pth'

    # 读取label-id映射关系记录文件
    lbl2id_map_path = os.path.join(base_data_dir, 'lbl2id_map.txt')
    lbl2id_map, id2lbl_map = load_lbl2id_map(lbl2id_map_path)

    # 统计数据集中出现的所有的label中包含字符最多的有多少字符，数据集构造gt信息需要用到
    train_lbl_path = os.path.join(base_data_dir, 'train_gt.txt')
    valid_lbl_path = os.path.join(base_data_dir, 'valid_gt.txt')
    train_max_label_len = statistics_max_len_label(train_lbl_path)
    valid_max_label_len = statistics_max_len_label(valid_lbl_path)
    sequence_len = max(train_max_label_len, valid_max_label_len)  # 数据集中字符数最多的一个case作为制作的gt的sequence_len

    # 构造 dataloader
    max_ratio = 8  # 图片预处理时 宽/高的最大值，不超过就保比例resize，超过会强行压缩
    train_dataset = RecognitionDataset(base_data_dir, lbl2id_map, sequence_len, max_ratio, 'train', pad=0)
    valid_dataset = RecognitionDataset(base_data_dir, lbl2id_map, sequence_len, max_ratio, 'valid', pad=0)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4)

    # build model
    # use transformer as ocr recognize model
    tgt_vocab = len(lbl2id_map.keys())
    d_model = 512
    ocr_model = make_ocr_model(tgt_vocab, n=5, d_model=d_model, d_ff=2048, h=8, dropout=0.1)
    ocr_model.to(device)

    # train prepare
    criterion = LabelSmoothing(size=tgt_vocab, padding_idx=0, smoothing=0.0)

    # choose a optimizer
    # optimizer = torch.optim.Adam(ocr_model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
    optimizer = torch.optim.SGD(ocr_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(nRofEpochs):
        logger.info(f"\nepoch {epoch}")

        logger.info("train...")
        ocr_model.train()
        # loss_compute = SimpleLossCompute(ocr_model.generator, criterion, model_opt)
        loss_compute = SimpleLossCompute(ocr_model.generator, criterion, optimizer)
        train_mean_loss = run_epoch(train_loader, ocr_model, loss_compute, device)

        if epoch % 10 == 0:
            logger.info("valid...")
            ocr_model.eval()
            valid_loss_compute = SimpleLossCompute(ocr_model.generator, criterion, None)
            valid_mean_loss = run_epoch(valid_loader, ocr_model, valid_loss_compute, device)
            logger.info(f"valid loss: {valid_mean_loss}")

    # save model
    torch.save(ocr_model.state_dict(), model_save_path)

    # 训练结束，使用贪心的解码方式推理训练集和验证集，统计正确率
    ocr_model.eval()
    logger.info("\n------------------------------------------------")
    logger.info("greedy decode train set")
    total_img_num = 0
    total_correct_num = 0
    for batch_idx, batch in enumerate(train_loader):
        img_input, encode_mask, decode_in, decode_out, decode_mask, nTokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)

        bs = img_input.shape[0]
        for i in range(bs):
            cur_img_input = img_input[i].unsqueeze(0)
            cur_encode_mask = encode_mask[i].unsqueeze(0)
            cur_decode_out = decode_out[i]

            predictResult = greedy_decode(ocr_model, cur_img_input, cur_encode_mask, max_len=sequence_len,
                                          start_symbol=1,
                                          end_symbol=2)
            predictResult = predictResult.cpu()

            is_correct = judge_is_correct(predictResult, cur_decode_out)
            total_correct_num += is_correct
            total_img_num += 1
            if not is_correct:
                # 预测错误的case进行打印
                logger.info("----")
                logger.info(cur_decode_out)
                logger.info(predictResult)
    total_correct_rate = total_correct_num / total_img_num * 100
    logger.info(f"total correct rate of train set: {total_correct_rate}%")

    logger.info("\n------------------------------------------------")
    logger.info("greedy decode valid set")
    total_img_num = 0
    total_correct_num = 0
    for batch_idx, batch in enumerate(valid_loader):
        img_input, encode_mask, decode_in, decode_out, decode_mask, nTokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)

        bs = img_input.shape[0]
        for i in range(bs):
            cur_img_input = img_input[i].unsqueeze(0)
            cur_encode_mask = encode_mask[i].unsqueeze(0)
            cur_decode_out = decode_out[i]

            predictResult = greedy_decode(ocr_model, cur_img_input, cur_encode_mask, max_len=sequence_len,
                                          start_symbol=1,
                                          end_symbol=2)
            predictResult = predictResult.cpu()

            is_correct = judge_is_correct(predictResult, cur_decode_out)
            total_correct_num += is_correct
            total_img_num += 1
            if not is_correct:
                # 预测错误的case进行打印
                logger.info("----")
                logger.info(cur_decode_out)
                logger.info(predictResult)
    total_correct_rate = total_correct_num / total_img_num * 100
    logger.info(f"total correct rate of valid set: {total_correct_rate}%")
