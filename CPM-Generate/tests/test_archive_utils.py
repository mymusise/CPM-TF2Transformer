"""Offline regression tests; no Torch, model weights or network required."""

import ast
import importlib.util
import io
import logging
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "archive_utils", str(ROOT / "model" / "archive_utils.py"))
archive_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_utils)


def make_archive(entries):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name, kind, content in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.mode = 0o755 if kind == tarfile.DIRTYPE else 0o644
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = content
            elif kind == tarfile.REGTYPE:
                member.size = len(content)
            archive.addfile(member, io.BytesIO(content) if kind == tarfile.REGTYPE else None)
    data.seek(0)
    return data


def loader_extraction_block():
    # Exercise the actual loader's extraction branch without importing Torch.
    tree = ast.parse((ROOT / "model" / "modeling.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "PreTrainedBertModel")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "from_pretrained")
    branch = next(n for n in method.body if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Call)
                  and isinstance(n.test.func, ast.Attribute)
                  and n.test.func.attr == "isdir")
    module = ast.Module(body=[branch], type_ignores=[])
    return compile(ast.fix_missing_locations(module), "modeling-extraction-branch", "exec")


class ArchiveExtractionTests(unittest.TestCase):
    def extract(self, data, target):
        with tarfile.open(fileobj=data, mode="r:gz") as archive:
            archive_utils.extract_model_archive(archive, target)

    def test_regular_files_directories_dot_and_repeated_files(self):
        with tempfile.TemporaryDirectory() as target:
            entries = [(".", tarfile.DIRTYPE, b""),
                       ("nested", tarfile.DIRTYPE, b""),
                       ("./bert_config.json", tarfile.REGTYPE, b"{}"),
                       ("nested/weights.bin", tarfile.REGTYPE, b"old"),
                       ("nested/weights.bin", tarfile.REGTYPE, b"new")]
            self.extract(make_archive(entries), target)
            self.assertEqual((Path(target) / "bert_config.json").read_bytes(), b"{}")
            self.assertEqual((Path(target) / "nested/weights.bin").read_bytes(), b"new")

    def test_reject_unsafe_members_before_writing_anything(self):
        entries = [("../sentinel", tarfile.REGTYPE, b"bad"),
                   ("nested/../../sentinel", tarfile.REGTYPE, b"bad"),
                   ("/absolute/sentinel", tarfile.REGTYPE, b"bad"),
                   ("C:/sentinel", tarfile.REGTYPE, b"bad"),
                   ("C:sentinel", tarfile.REGTYPE, b"bad"),
                   ("..\\sentinel", tarfile.REGTYPE, b"bad"),
                   ("link", tarfile.SYMTYPE, "../sentinel"),
                   ("hardlink", tarfile.LNKTYPE, "../sentinel"),
                   ("pipe", tarfile.FIFOTYPE, b""),
                   ("device", tarfile.CHRTYPE, b"")]
        for entry in entries:
            with self.subTest(entry=entry[0]), tempfile.TemporaryDirectory() as target:
                with self.assertRaises(ValueError):
                    self.extract(make_archive([("valid", tarfile.REGTYPE, b"ok"), entry]), target)
                self.assertEqual(os.listdir(target), [])

    def test_reject_existing_symlink_escape(self):
        with tempfile.TemporaryDirectory() as target, tempfile.TemporaryDirectory() as outside:
            try:
                os.symlink(outside, os.path.join(target, "nested"))
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaises(ValueError):
                self.extract(make_archive([("nested/sentinel", tarfile.REGTYPE, b"bad")]), target)
            self.assertEqual(os.listdir(outside), [])

    def test_loader_integration_legitimate_archive_and_directory(self):
        code = loader_extraction_block()
        with tempfile.TemporaryDirectory() as work:
            archive_path = os.path.join(work, "model.tar.gz")
            Path(archive_path).write_bytes(make_archive([
                ("bert_config.json", tarfile.REGTYPE, b"{}"),
                ("pytorch_model.bin", tarfile.REGTYPE, b"not-a-real-model")]).getvalue())
            env = self.loader_environment(archive_path)
            exec(code, env)
            try:
                self.assertEqual((Path(env["serialization_dir"]) / "bert_config.json").read_bytes(), b"{}")
                self.assertEqual((Path(env["serialization_dir"]) / "pytorch_model.bin").read_bytes(), b"not-a-real-model")
            finally:
                shutil.rmtree(env["tempdir"])
            env = self.loader_environment(work)
            exec(code, env)
            self.assertEqual(env["serialization_dir"], work)

    def loader_environment(self, archive_path):
        return {"os": os, "tarfile": tarfile, "tempfile": tempfile,
                "shutil": shutil, "logger": logging.getLogger(__name__),
                "resolved_archive_file": archive_path,
                "extract_model_archive": archive_utils.extract_model_archive}

    def test_loader_cleans_rejected_and_corrupt_archives(self):
        code = loader_extraction_block()
        with tempfile.TemporaryDirectory() as work:
            archive_path = os.path.join(work, "model.tar.gz")
            for content, error in [(make_archive([("../sentinel", tarfile.REGTYPE, b"bad")]).getvalue(), ValueError),
                                   (b"not a tar archive", tarfile.TarError)]:
                Path(archive_path).write_bytes(content)
                temp = os.path.join(work, "extraction")
                os.mkdir(temp)
                with patch.object(tempfile, "mkdtemp", return_value=temp):
                    with self.assertRaises(error):
                        exec(code, self.loader_environment(archive_path))
                self.assertFalse(os.path.exists(temp))
                self.assertFalse(os.path.exists(os.path.join(work, "sentinel")))


if __name__ == "__main__":
    unittest.main()
