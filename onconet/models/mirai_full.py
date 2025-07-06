import logging
import os
import pickle
import tempfile
import traceback
from typing import List, BinaryIO
import warnings
import zipfile

import numpy as np
import pydicom
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)

import onconet.transformers.factory as transformer_factory
import onconet.models.calibrator
import onconet.utils.dicom
from onconet import __version__ as onconet_version
from onconet.models.factory import load_model, RegisterModel, get_model_by_name
from onconet.models.factory import get_model
from onconet.transformers.basic import ComposeTrans
from onconet.utils import parsing
from onconet.utils.logging_utils import get_logger


@RegisterModel("mirai_full")
class MiraiFull(nn.Module):

    def __init__(self, args):
        super(MiraiFull, self).__init__()
        self.args = args
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
        else:
            self.image_encoder = get_model_by_name('custom_resnet', False, args)

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim
        if args.transformer_snapshot is not None:
            self.transformer = load_model(args.transformer_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = get_model_by_name('transformer', False, args)
        args.img_only_dim = self.transformer.args.transfomer_hidden_dim

    def forward(self, x, risk_factors=None, batch=None):
        B, C, N, H, W = x.size()
        x = x.transpose(1,2).contiguous().view(B*N, C, H, W)
        risk_factors_per_img =  (lambda N, risk_factors: [factor.expand( [N, *factor.size()]).contiguous().view([-1, factor.size()[-1]]).contiguous() for factor in risk_factors])(N, risk_factors) if risk_factors is not None else None
        _, img_x, _ = self.image_encoder(x, risk_factors_per_img, batch)
        img_x = img_x.view(B, N, -1)
        img_x = img_x[:,:,: self.image_repr_dim]
        logit, transformer_hidden, activ_dict = self.transformer(img_x, risk_factors, batch)
        return logit, transformer_hidden, activ_dict


def download_file(url, destination):
    import urllib.request

    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as e:
        get_logger().error(f"An error occurred while downloading from {url} to {destination}: {e}")
        raise e


def _torch_set_num_threads(threads) -> int:
    """
    Set the number of CPU threads for torch to use.
    Set to a negative number for no-op.
    Set to 0 for the number of CPUs.
    """
    if threads < 0:
        return torch.get_num_threads()
    if threads is None or threads == 0:
        # I've never seen a benefit to going higher than 8 and sometimes there is a big slowdown
        threads = min(8, os.cpu_count())

    torch.set_num_threads(threads)
    return torch.get_num_threads()


class MiraiModel:
    """
    Represents a trained Mirai model. Useful for predictions on individual exams.
    """
    def __init__(self, config_obj):
        super().__init__()
        self.args = self.sanitize_paths(config_obj)
        self.__version__ = onconet_version

    def load_model(self):
        logger = get_logger()
        logger.debug("Loading model...")
        self.args.cuda = self.args.cuda and torch.cuda.is_available()

        self.download_if_needed(self.args)
        if self.args.model_name == 'mirai_full':
            model = get_model(self.args)
        else:
            model = torch.load(self.args.snapshot, map_location='cpu')

        # Unpack models that were trained as data parallel
        if isinstance(model, nn.DataParallel):
            model = model.module

        # Add use precomputed hiddens for models trained before it was introduced.
        # Assumes a resnet WHybase backbone
        try:
            model._model.args.use_precomputed_hiddens = self.args.use_precomputed_hiddens
            model._model.args.cuda = self.args.cuda
        except Exception as e:
            logger.debug("Exception caught, skipping precomputed hiddens")
            pass

        return model

    def load_calibrator(self):
        get_logger().debug("Loading calibrator...")

        # Load calibrator if desired
        if self.args.calibrator_path is not None:
            with open(self.args.calibrator_path, 'rb') as infi:
                calibrator = pickle.load(infi)
        else:
            calibrator = None

        return calibrator

    def process_image_joint(self, batch, model, calibrator, risk_factor_vector=None):
        logger = get_logger()
        logger.debug("Getting predictions...")

        if self.args.cuda:
            device = get_default_device()
            logger.debug(f"Inference with {device}")
            for obj in [model, model.transformer]:
                obj.to(device)
            for key, val in batch.items():
                batch[key] = val.to(device)
        else:
            model = model.cpu()
            logger.debug("Inference with CPU")

        risk_factors = autograd.Variable(risk_factor_vector.unsqueeze(0)) if risk_factor_vector is not None else None

        logit, _, _ = model(batch['x'], risk_factors, batch)
        probs = F.sigmoid(logit).cpu().data.numpy()
        pred_y = np.zeros(probs.shape[1])

        if calibrator is not None:
            logger.debug("Raw probs: {}".format(probs))

            for i in calibrator.keys():
                pred_y[i] = calibrator[i].predict_proba(probs[0, i].reshape(-1, 1)).flatten()[1]

        return pred_y.tolist()

    def process_exam(self, images, risk_factor_vector):
        # Below lines commented for model to accept <4 dicoms as well
        # if len(images) != 4:
        #     raise ValueError(f"Require exactly 4 images, instead we got {len(images)}")

        logger = get_logger()
        logger.debug(f"Processing images...")

        test_image_transformers = parsing.parse_transformers(self.args.test_image_transformers)
        test_tensor_transformers = parsing.parse_transformers(self.args.test_tensor_transformers)
        test_transformers = transformer_factory.get_transformers(test_image_transformers, test_tensor_transformers, self.args)
        transforms = ComposeTrans(test_transformers)

        batch = self.collate_batch(images, transforms)
        model = self.load_model()
        calibrator = self.load_calibrator()

        y = self.process_image_joint(batch, model, calibrator, risk_factor_vector)

        return y

    def collate_batch(self, images, transforms):
        get_logger().debug("Collating batches...")

        batch = {}
        batch['side_seq'] = torch.cat([torch.tensor(b['side_seq']).unsqueeze(0) for b in images], dim=0).unsqueeze(0)
        batch['view_seq'] = torch.cat([torch.tensor(b['view_seq']).unsqueeze(0) for b in images], dim=0).unsqueeze(0)
        batch['time_seq'] = torch.zeros_like(batch['view_seq'])

        batch['x'] = torch.cat(
            (lambda imgs: [transforms(b['x']).unsqueeze(0) for b in imgs])(images), dim=0
        ).unsqueeze(0).transpose(1, 2)

        return batch

    def run_model(self, dicom_files: List[BinaryIO], payload=None):
        logger = get_logger()
        _torch_set_num_threads(getattr(self.args, 'threads', 0))
        if payload is None:
            payload = dict()

        # Below commented out for PNG input acceptance
        # dcmread_force = payload.get("dcmread_force", False)
        # dcmtk_installed = onconet.utils.dicom.is_dcmtk_installed()
        # use_dcmtk = payload.get("dcmtk", True) and dcmtk_installed
        # if use_dcmtk:
        #     logger.info('Using dcmtk')
        # else:
        #     logger.info('Using pydicom')

        images = []
        # dicom_info = {}
        # the entire below section is modified to enable png predictions
        if isinstance(input_files[0], str):
            for file_path in input_files:
                try:
                    original_filename = os.path.basename(file_path)
                    if not ('_CC.' in original_filename.upper() or '_MLO.' in original_filename.upper()):
                        raise ValueError(f"Filename must contain view/side like R_CC.png. Got: {original_filename}")

                    with open(file_path, 'rb') as f:
                        file_content = f.read()

                    # Create temp file with original name structure preserved
                    with tempfile.NamedTemporaryFile(
                        prefix='mirai_',
                        suffix=f'_{original_filename}',
                        dir='/tmp'
                    ) as tmp_png:
                        tmp_png.write(file_content)
                        tmp_png.flush()

                        view, side = onconet.utils.dicom.get_png_info(tmp_png.name)
                        image = onconet.utils.dicom.png_to_image(tmp_png.name)
                        images.append({'x': image, 'side_seq': side, 'view_seq': view})

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    raise
        else:
            for i, input_file in enumerate(input_files):
                try:
                    input_file.seek(0)
                    file_start = input_file.read(132)
                    input_file.seek(0)

                    # Check for PNG first (simpler signature)
                    if file_start.startswith(b'\x89PNG'):
                        logger.info("Processing PNG file")

                        # Generate a filename based on position since we don't have the original
                        positions = ['L_CC', 'R_CC', 'L_MLO', 'R_MLO']
                        if i < len(positions):
                            original_filename = f"{positions[i]}.png"
                        else:
                            raise ValueError("Too many input files - expected exactly 4")

                        # Create temp file with position-based name
                        with tempfile.NamedTemporaryFile(
                            prefix='mirai_',
                            suffix=f'_{original_filename}',
                            dir='/tmp'
                        ) as tmp_png:
                            tmp_png.write(input_file.read())
                            tmp_png.flush()

                            view, side = onconet.utils.dicom.get_png_info(tmp_png.name)
                            image = onconet.utils.dicom.png_to_image(tmp_png.name)
                            images.append({'x': image, 'side_seq': side, 'view_seq': view})

                        # Rest of the DICOM handling remains the same...
                    elif b'DICM' in file_start[:4] or (len(file_start) > 132 and b'DICM' in file_start[128:132]):
                        # Handle DICOM file (unchanged from original)
                        dcmread_force = payload.get("dcmread_force", False)
                        dcmtk_installed = onconet.utils.dicom.is_dcmtk_installed()
                        use_dcmtk = payload.get("dcmtk", True) and dcmtk_installed

                        if use_dcmtk:
                            logger.info('Using dcmtk for DICOM')
                            with tempfile.NamedTemporaryFile(suffix='.dcm') as dcm_file:
                                dcm_file.write(input_file.read())
                                dcm_file.flush()
                                with tempfile.NamedTemporaryFile(suffix='.png') as img_file:
                                    image = onconet.utils.dicom.dicom_to_image_dcmtk(dcm_file.name, img_file.name)
                                    tmp_dcm = pydicom.dcmread(dcm_file.name, stop_before_pixels=True)
                                    view, side = onconet.utils.dicom.get_dicom_info(tmp_dcm)
                                    images.append({'x': image, 'side_seq': side, 'view_seq': view})
                        else:
                            logger.info('Using pydicom for DICOM')
                            try:
                                dicom = pydicom.dcmread(input_file, force=dcmread_force)
                                window_method = payload.get("window_method", "minmax")
                                image = onconet.utils.dicom.dicom_to_arr(dicom, window_method=window_method, pillow=True)
                                view, side = onconet.utils.dicom.get_dicom_info(dicom)
                                images.append({'x': image, 'side_seq': side, 'view_seq': view})
                            except Exception as e:
                                logger.error(f"Error reading DICOM file: {e}")
                                raise
                    else:
                        try:
                            logger.warning("File doesn't have standard DICOM signature, attempting to read anyway")
                            dicom = pydicom.dcmread(input_file, force=True)
                            window_method = payload.get("window_method", "minmax")
                            image = onconet.utils.dicom.dicom_to_arr(dicom, window_method=window_method, pillow=True)
                            view, side = onconet.utils.dicom.get_dicom_info(dicom)
                            images.append({'x': image, 'side_seq': side, 'view_seq': view})
                        except Exception as e:
                            raise ValueError(f"Could not read file as either DICOM or PNG: {e}")

                except Exception as e:
                    logger.warning(f"{type(e).__name__}: {e}")
                    logger.warning(f"{traceback.format_exc()}")

            logger.info(f"Total images processed: {len(images)}")
            for idx, img_entry in enumerate(images):
                img = img_entry['x']
                logger.info(f"Image {idx + 1}: size = {img.size}, mode = {img.mode}")

            risk_factor_vector = None
            y = self.process_exam(images, risk_factor_vector)
            logger.debug(f'Raw Predictions: {y}')

            y = {'Year {}'.format(i+1): round(p, 4) for i, p in enumerate(y)}
            report = {'predictions': y, 'modelVersion': self.__version__}

            return report

    @staticmethod
    def sanitize_paths(args):
        path_keys = ["img_encoder_snapshot", "transformer_snapshot", "calibrator_path"]
        for key in path_keys:
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(args, key, os.path.expanduser(getattr(args, key)))
        return args

    @staticmethod
    def download_if_needed(args, cache_dir='./.cache'):
        args = MiraiModel.sanitize_paths(args)
        if args.model_name == 'mirai_full':
            if os.path.exists(args.img_encoder_snapshot) and os.path.exists(args.transformer_snapshot):
                return
        else:
            if os.path.exists(args.snapshot):
                return

        if getattr(args, 'remote_snapshot_uri', None) is None:
            return

        get_logger().info(f"Local models not found, downloading snapshot from remote URI: {args.remote_snapshot_uri}")
        os.makedirs(cache_dir, exist_ok=True)
        tmp_zip_path = os.path.join(cache_dir, "snapshots.zip")
        if not os.path.exists(tmp_zip_path):
            download_file(args.remote_snapshot_uri, tmp_zip_path)

        dest_dir = os.path.dirname(args.img_encoder_snapshot) if args.model_name == 'mirai_full' else os.path.dirname(args.snapshot)

        # Unzip file
        get_logger().info(f"Saving models to {dest_dir}")
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        os.remove(tmp_zip_path)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # Not all operations implemented in MPS yet
        use_mps = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
        if use_mps:
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')
