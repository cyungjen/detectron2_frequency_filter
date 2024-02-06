import numpy as np
import cv2
from matplotlib import pyplot as plt

from collections import Counter, defaultdict
import heapq
import time
import os
import logging

'''huffman decode'''
def invert_codebook(codebook):
    return {v: k for k, v in codebook.items()}

def huffman_decode(encoded_data, inverted_codebook):
    decoded_data = []
    current_code = ""
    for bit in encoded_data:
        current_code += bit
        if current_code in inverted_codebook:
            decoded_character = inverted_codebook[current_code]
            decoded_data.append(int(decoded_character)) 
            current_code = ""
    return decoded_data

def decode_huffman(encoded_data, codebook):
    inverted_codebook = invert_codebook(codebook)
    return huffman_decode(encoded_data, inverted_codebook)


'''huffman encode'''
def calculate_frequency(data):
    frequency = Counter(data)
    return frequency
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    if not frequency:
        # 如果頻率字典為空，則返回 None 或適當的默認值
        # logging.info(f'none')
        return None

    priority_queue = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)

    if len(priority_queue) == 1:
        # 如果只有一個元素，創建一個新節點作為根節點
        return Node(None, priority_queue[0].freq, left=priority_queue[0])

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)

    return priority_queue[0]

def build_codes(node, prefix="", codebook=defaultdict()):
    if node is not None:
        if node.char is not None:
            codebook[node.char] = prefix
        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data, codebook):
    return ''.join(codebook[char] for char in data)

def huffman_coding(data):
    frequency = calculate_frequency(data)
    root = build_huffman_tree(frequency)
    codebook = build_codes(root)
    encoded_data = huffman_encode(data, codebook)
    return encoded_data, codebook

def pad_to_8_multiple(image):
    """
    填充圖像至8的倍數。
    :param image: input image
    :return: padded image
    """
    # image = np.array(image)
    h, w = image.shape[:2]
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    
    return cv2.copyMakeBorder(image, 0, h_pad, 0, w_pad, cv2.BORDER_REPLICATE),h,w


def matrix_to_triplets(matrix):
    """
    將矩陣轉為triplet。
    :param matrix: 輸入的 NumPy 矩陣
    :return: 一個列表，包含矩陣中所有非零元素的 (行, 列, 值) 三元組
    """
    triplets = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != 0:
                triplets.append((i, j, matrix[i, j]))
    return np.array(triplets)

def array_to_matrix(array, rows = 8, cols = 8):
    """
    將包含三元組的數組轉換回矩陣。
    :param array: 包含 (行, 列, 值) 三元組的 NumPy 數組
    :param rows: 目標矩陣的行數
    :param cols: 目標矩陣的列數
    :return: 轉換後的 NumPy 矩陣
    """
    matrix = np.zeros((rows, cols), dtype=float)
    try:
        for i, j, val in array:
            matrix[i, j] = val
        return matrix
    except:
        logging.info(f'array = {array}')




def process_image_block_F(img,w,h,qf,folder_name,writeimg):
    """
    Image preprocess. dct -> frequency filter -> Quantize -> triplets 
    -> huffman -> run length coding -> decoding...
    """
    global Qtable_Y 
    global Qtable_CbCr
    global bitcnt_all
    block_size = 8
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    '''Do quantize and filter in the same time.'''
    #8*8 quantize matrix
    filter_Y = np.array([
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    filter_CbCr = np.array([
        [1, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    img_idct = []
    bitcnt = 0
    
        
    for i in range(3):
        img_float32 = np.float32(img[:,:,i])
        idct = np.zeros((h, w), dtype=float)
        # 對每塊block做dct
        for j in range(0, h, block_size):
            for k in range(0, w, block_size):
                block = img_float32[j:j+block_size, k:k+block_size]
                
                #dct
                block = cv2.dct(block.astype(float))

                #frequency filter
                if i == 0: 
                    block = block*filter_Y
                else:
                    block = block*filter_CbCr

                #quantize 
                if i == 0: 
                    block = np.round(block/Qtable_Y)
                else:
                    block = np.round(block/Qtable_CbCr)

                #triplet 
                block = block.astype(int)
                block_triplets = matrix_to_triplets(block)

                flattened_data = block_triplets.flatten()
                flattened_data_str = [str(x) for x in flattened_data]
                if len(set(flattened_data_str)) == 1:
                    decoded_data_int = [int(flattened_data_str[0])] * len(flattened_data_str)
                else:
                    # Huffman coding
                    encoded_data, codebook = huffman_coding(flattened_data_str)
                    # print('len(encoded_data)=',len(encoded_data))
                    bitcnt += len(encoded_data)
                    # Huffman decoding
                    decoded_data = decode_huffman(encoded_data, codebook)
                    # logging.info(f'len(encoded_data)={len(encoded_data)}')
                    # logging.info(f'encoded_data={encoded_data}')
                    # logging.info(f'decoded_data={decoded_data}')
                    codebook.clear()  # 清除字典
                    decoded_data_int = [int(x) for x in decoded_data]
                    

                decoded_data_2d = []
                decoded_data_2d = [decoded_data_int[i:i+3] for i in range(0, len(decoded_data_int), 3)]

                matrix = array_to_matrix(decoded_data_2d)
                matrix = np.array(matrix)
                #dequantize
                if i == 0: 
                    matrix = matrix*Qtable_Y
                else:
                    matrix = matrix*Qtable_CbCr
                #idct
                block_idct = cv2.idct(matrix)
            
                idct[j:j+block_size, k:k+block_size] = block_idct
                

        img_idct.append(idct)
    bitcnt_all += bitcnt
    # print(bitcnt_all)
    img_reconstructed = cv2.merge(img_idct)
    #確保數據類型正確並且值在 0 到 255 的範圍內
    img_reconstructed = np.clip(img_reconstructed, 0, 255)
    img_reconstructed = img_reconstructed.astype(np.uint8)
    img_reconstructed = cv2.cvtColor(img_reconstructed, cv2.COLOR_YUV2BGR)
    if writeimg == 1:
        cv2.imwrite(f"image/{folder_name}_img_{qf}.png", img_reconstructed)
    writeimg = 0
    return bitcnt_all


np.set_printoptions(threshold=np.inf)
'''
This file is for QF-bitrate test
Chose a dataset you wanna test 
'''
root_folder_path = 'image_416x240'
logging.basicConfig(filename=f'log/{root_folder_path}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

#decide QF sequence by video size
qf_sequence_dict = {
    'image_416x240'  : [2,6,10,19,32,50,60,80,99],
    'image_832x480'  : [2,6,10,19,32,50,60],
    'Image_1920x1080': [0.1,1,2,5,10,20,30,40],
    'image_2560x1600': [0.1,1,2,5,10,20,30,40]
}
sequence = qf_sequence_dict[root_folder_path]

folder_path = os.listdir(root_folder_path)

for path in folder_path:
    images_names = os.listdir(f'{root_folder_path}/{path}/imgs')
    #find fps
    folder_cls = path.split('_')
    fps = folder_cls[2]

    #find frame num
    frame_num_dict = {
        'Traffic': 33,
        'ParkScene': 33,
        'Cactus': 97,
        'BasketballDrive': 97,
        'BQTerrace': 129,
        'BQMall': 129,
        'PartyScene': 97,
        'BasketballDrill': 97,
        'RaceHorses': 65,
        'BQSquare': 129,
        'BlowingBubbles': 97,
        'BasketballPass': 97
    }
    frame_num = frame_num_dict[folder_cls[0]]
    # print(fps,frame_num)
    for qf in sequence:
        Qtable_Y = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        Qtable_CbCr = np.array([
            [1,  18, 24, 47, 99, 99, 99, 99],
            [1,  21, 26, 66, 99, 99, 99, 99],
            [2,  26, 56, 99, 99, 99, 99, 99],
            [4,  66, 99, 99, 99, 99, 99, 99],
            [9,  99, 99, 99, 99, 99, 99, 99],
            [9,  99, 99, 99, 99, 99, 99, 99],
            [9,  99, 99, 99, 99, 99, 99, 99],
            [9,  99, 99, 99, 99, 99, 99, 99]
        ])

        if qf>=50:
            Qtable_Y = np.maximum(np.floor((2-qf/50)*Qtable_Y+0.5),1)
            Qtable_CbCr = np.maximum(np.floor((2-qf/50)*Qtable_CbCr+0.5),1)
        else:
            Qtable_Y = np.floor(50/qf*Qtable_Y+0.5)
            Qtable_CbCr = np.floor(50/qf*Qtable_CbCr+0.5)

        writeimg = 1
        bitcnt_all = 0
        # 逐個讀取圖像
        for image_name in images_names:
            # 讀取完整的文件路徑
            image_path = os.path.join(f'{root_folder_path}/{path}/imgs', image_name)
            # 讀取圖像
            img = cv2.imread(image_path)
            padded_image,w,h = pad_to_8_multiple(img)
            bits = process_image_block_F(padded_image,h,w,qf,path,writeimg)
            writeimg = 0
        print('bits=',bits)
        bitrate = bits/frame_num*int(fps)/1000000
        logging.info(f'{path} QP={qf} bitrate={bitrate}M')
