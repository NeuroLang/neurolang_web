import sys
import os
from setuptools import setup
from setuptools.command.develop import develop

pjoin = os.path.join


class DevelopCmd(develop):
    prefix_targets = [
        ("nbconvert/templates", "neurolang"),
    ]

    def run(self):
        target_dir = os.path.join(sys.prefix, "share", "jupyter")
        target_dir = os.path.join(target_dir)

        for prefix_target, name in self.prefix_targets:
            source = os.path.join("share", "jupyter", prefix_target, name)
            target = os.path.join(target_dir, prefix_target, name)
            target_subdir = os.path.dirname(target)
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)
            rel_source = os.path.relpath(
                os.path.abspath(source), os.path.abspath(target_subdir)
            )
            try:
                os.remove(target)
            except:
                pass
            print(rel_source, "->", target)
            os.symlink(rel_source, target)

        super(DevelopCmd, self).run()


data_files = []
# Add all the templates
for (dirpath, dirnames, filenames) in os.walk("share/jupyter/"):
    if filenames:
        data_files.append(
            (
                dirpath,
                [os.path.join(dirpath, filename) for filename in filenames],
            )
        )

setup(
    data_files=data_files,
    include_package_data=True,
    keywords=["ipython", "jupyter", "widgets", "voila"],
    cmdclass={
        "develop": DevelopCmd,
    },
)
