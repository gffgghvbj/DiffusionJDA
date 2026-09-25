from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import torchreid
import numpy as np
import sys
from .fast_reid_adopter import FastReID


class EmbeddingComputer:
    def __init__(self, dataset, test_dataset, grid_off, max_batch=1024):
        self.model = None
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.crop_size = (128, 384)
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        self.grid_off = grid_off
        self.max_batch = max_batch

        # Only used for the general ReID model (not FastReID)
        self.normalize = False

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def get_horizontal_split_patches(self, image, bbox, tag, idx, viz=False):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.shape[2:]

        bbox = np.array(bbox)
        bbox = bbox.astype(np.int)
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
            # Faulty Patch Correction
            bbox[0] = np.clip(bbox[0], 0, None)
            bbox[1] = np.clip(bbox[1], 0, None)
            bbox[2] = np.clip(bbox[2], 0, image.shape[1])
            bbox[3] = np.clip(bbox[3], 0, image.shape[0])

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        ### TODO - Write a generalized split logic
        split_boxes = [
            [x1, y1, x1 + w, y1 + h / 3],
            [x1, y1 + h / 3, x1 + w, y1 + (2 / 3) * h],
            [x1, y1 + (2 / 3) * h, x1 + w, y1 + h],
        ]

        split_boxes = np.array(split_boxes, dtype="int")
        patches = []
        # breakpoint()
        for ix, patch_coords in enumerate(split_boxes):
            if isinstance(image, np.ndarray):
                im1 = image[patch_coords[1]: patch_coords[3], patch_coords[0]: patch_coords[2], :]

                if viz:  ## TODO - change it from torch tensor to numpy array
                    dirs = "./viz/{}/{}".format(tag.split(":")[0], tag.split(":")[1])
                    Path(dirs).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(dirs, "{}_{}.png".format(idx, ix)),
                        im1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255,
                    )
                patch = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                patch = cv2.resize(patch, self.crop_size, interpolation=cv2.INTER_LINEAR)
                patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
                patch = patch.unsqueeze(0)
                # print("test ", patch.shape)
                patches.append(patch)
            else:
                im1 = image[:, :, patch_coords[1]: patch_coords[3], patch_coords[0]: patch_coords[2]]
                patch = torchvision.transforms.functional.resize(im1, (256, 128))
                patches.append(patch)

        patches = torch.cat(patches, dim=0)

        # print("Patches shape ", patches.shape)
        # patches = np.array(patches)
        # print("ALL SPLIT PATCHES SHAPE - ", patches.shape)

        return patches

    def compute_embedding(self, img, bbox, tag, det=True):

        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # bbox = shrink_boxes(bbox.cpu())
        bbox = np.array(bbox.cpu(), dtype=np.int32) if not isinstance(bbox, np.ndarray) \
            else np.array(bbox, dtype=np.int32)

        if self.model is None:
            self.initialize_model()

        # Generate all of the patches
        crops = []
        if self.grid_off:
            # Basic embeddings
            h, w = img.shape[:2]
            results = np.round(bbox).astype(np.int32)
            results[:, 0] = results[:, 0].clip(0, w)
            results[:, 1] = results[:, 1].clip(0, h)
            results[:, 2] = results[:, 2].clip(0, w)
            results[:, 3] = results[:, 3].clip(0, h)

            for p in results:
                # 检查框的面积是否为0
                if p[2] - p[0] == 0 or p[3] - p[1] == 0:
                    # 如果宽度或高度为 0，返回全零向量 (1, 520)
                    # print(f"Detected zero-area bounding box: {p}, returning zero embedding.")
                    zero_crop = torch.zeros((1, 3, self.crop_size[1], self.crop_size[0]), dtype=torch.float32)
                    crops.append(zero_crop)
                    continue

                crop = img[p[1]: p[3], p[0]: p[2]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

                if self.normalize:
                    crop /= 255
                    crop -= np.array((0.485, 0.456, 0.406))
                    crop /= np.array((0.229, 0.224, 0.225))
                crop = torch.as_tensor(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
                crops.append(crop)
        else:
            # Grid patch embeddings
            for idx, box in enumerate(bbox):
                crop = self.get_horizontal_split_patches(img, box, tag, idx)
                crops.append(crop)

        # 如果有非空裁剪图像，继续生成特征
        crops = torch.cat(crops, dim=0)
        # print(crops.shape)
        embs = []
        for idx in range(0, len(crops), self.max_batch):
            batch_crops = crops[idx: idx + self.max_batch]
            # if torch.all(batch_crops == 0):
            #     embs.extend(torch.zeros((1, 2048)).cuda())
            #     continue
            batch_crops = batch_crops.cuda()
            with torch.no_grad():
                batch_embs = self.model(batch_crops)
            embs.extend(batch_embs)
        embs = torch.stack(embs)
        embs = torch.nn.functional.normalize(embs, dim=-1)

        if not self.grid_off:
            embs = embs.reshape(bbox.shape[0], -1, embs.shape[-1])
        # embs = embs.cpu().numpy()

        if det:
            self.cache[tag] = embs

        return embs

    def initialize_model(self):
        if self.dataset == "mot17":
            if self.test_dataset:
                path = "/media/cvlab1045/D1/sp/code/DiffusionTrack-main/diffusion/models/external/weights/mot17_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == "mot20":
            if self.test_dataset:
                path = "/media/cvlab1045/D1/lzj/Deep-OC-SORT-main/external/weights/mot20_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == "dance":
            path = "/media/cvlab1045/D1/lzj/Deep-OC-SORT-main/external/weights/dance_sbs_S50.pth"
        else:
            raise RuntimeError("Need the path for a new ReID model.")

        model = FastReID(path)
        model.eval()
        model.cuda()
        model.half()
        self.model = model
        print("reid_model loading finished")

    def _get_general_model(self):
        """Used for the half-val for MOT17/20.

        The MOT17/20 SBS models are trained over the half-val we
        evaluate on as well. Instead we use a different model for
        validation.
        """
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)
        sd = torch.load("/media/cvlab1045/D1/lzj/Deep-OC-SORT-main/external/weights/osnet_ain_ms_d_c.pth.tar")[
            "state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        self.model = model
        self.crop_size = (128, 256)
        self.normalize = True

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)


if __name__ == '__main__':
    # Initialize the EmbeddingComputer
    embedding_computer = EmbeddingComputer(dataset='mot17', test_dataset=True, grid_off=True)

    # Load your image (replace 'your_image_path.jpg' with the actual image path)
    image = cv2.imread('/media/cvlab1045/D1/lzj/Deep-OC-SORT-main/data/mot/train/MOT17-02-DPM/img1/000001.jpg')

    # Define your bounding boxes (replace with your actual bounding boxes)
    # Example: 5 bounding boxes in the format [x1, y1, x2, y2]
    bounding_boxes = np.array([
        [50, 50, 150, 200],
        [200, 80, 300, 230],
        [350, 100, 450, 250],
        [500, 120, 600, 270],
        [650, 140, 750, 290]
    ])
    print(bounding_boxes.shape)
    # Compute the embeddings
    embeddings = embedding_computer.compute_embedding(image, bounding_boxes, tag='example_tag')

    # Print the embeddings
    print(embeddings.shape)


def shrink_boxes(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 计算中心点（保持不变）
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 原始宽高
    width = x2 - x1
    height = y2 - y1

    # 新宽高（减少1/10，即原尺寸的0.9倍）
    new_width = width * 0.9
    new_height = height * 0.9

    # 计算新坐标
    new_x1 = center_x - new_width / 2
    new_x2 = center_x + new_width / 2
    new_y1 = center_y - new_height / 2
    new_y2 = center_y + new_height / 2

    return torch.tensor(np.stack([new_x1, new_y1, new_x2, new_y2], axis=1))
