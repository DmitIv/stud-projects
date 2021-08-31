import subprocess
import sys
import os
import pathlib
from typing import List, Union
from functools import partial

executable_builder = "/usr/bin/xelatex"
path_to_tex_file = "/home/dmitri/PycharmProjects/BachelorDiploma/thesis/presentation/presentation.tex"
path_to_slides_dir = "/home/dmitri/PycharmProjects/BachelorDiploma/thesis/presentation/slides"

__path_to_temp_tex_file__ = path_to_tex_file.replace("presentation.tex", "tmp_presentation.tex")


def collect_filenames(path_to: str, target_extends: Union[List[str], str]) -> List[str]:
    slides_dir: pathlib.Path = pathlib.Path(path_to)
    if not slides_dir.exists():
        raise RuntimeWarning("No such directory")
    if type(target_extends) == str:
        target_extends = [target_extends]
    result: List[str] = []
    for file in slides_dir.iterdir():
        if file.is_file() and file.suffix in target_extends:
            result.append(file.stem)
    result.sort()
    return result


collect_tex_files = partial(collect_filenames, target_extends=".tex")


def document_part(path_to_slides: str) -> List[str]:
    doc_part: List[str] = []
    padding: str = "    "
    doc_part.append('\n')
    doc_part.append(r"\begin{document}" + '\n')
    doc_part.append(padding + r"\maketitle" + '\n')
    for slide_name in collect_tex_files(path_to_slides):
        include_value: str = os.path.join(path_to_slides.split(r"/")[-1], slide_name)
        include_string: str = r"\include{" + include_value + r"}" + '\n'
        doc_part.append(padding + include_string)
    doc_part.append(r"\end{document}" + '\n')
    return doc_part


def extend_with_slides(path_to_tex: str, path_to_slides: str, path_to_new_tex: str) -> None:
    old_tex: pathlib.Path = pathlib.Path(path_to_tex)
    if not old_tex.exists():
        raise RuntimeError("No such tex file")

    new_tex: pathlib.Path = pathlib.Path(path_to_new_tex)
    if new_tex.exists():
        print("Tex file will be overwrote")

    with old_tex.open(mode='r') as old_tex_file:
        tex_body: List[str] = old_tex_file.readlines()

    with new_tex.open(mode='w') as new_tex_file:
        new_tex_file.writelines(tex_body + document_part(path_to_slides))


def main() -> None:
    extend_with_slides(path_to_tex_file, path_to_slides_dir, __path_to_temp_tex_file__)

    cmd: str = "{0} {1}; mv {2} {3}; rm -f {4}".format(
        executable_builder,
        __path_to_temp_tex_file__,
        __path_to_temp_tex_file__.replace(".tex", ".pdf"),
        path_to_tex_file.replace(".tex", ".pdf"),
        __path_to_temp_tex_file__.replace(".tex", ".*"),
    )
    print("Execute: ", cmd)
    _ = subprocess.run(
        args=[cmd],
        shell=True,
        stdout=sys.stdout, stderr=sys.stderr,
        encoding='ascii', input='R'
    )


if __name__ == "__main__":
    main()
