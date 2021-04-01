# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from setuptools import find_packages, setup


VERSION = '0.1.7'
DESCRIPTION = 'diffwave'
AUTHOR = 'LMNT, Inc.'
AUTHOR_EMAIL = 'github@lmnt.com'
URL = 'https://www.lmnt.com'
LICENSE = 'Apache 2.0'
KEYWORDS = ['diffwave machine learning neural vocoder tts speech']
CLASSIFIERS = [
  'Development Status :: 4 - Beta',
  'Intended Audience :: Developers',
  'Intended Audience :: Education',
  'Intended Audience :: Science/Research',
  'License :: OSI Approved :: Apache Software License',
  'Programming Language :: Python :: 3.5',
  'Programming Language :: Python :: 3.6',
  'Programming Language :: Python :: 3.7',
  'Programming Language :: Python :: 3.8',
  'Topic :: Scientific/Engineering :: Mathematics',
  'Topic :: Software Development :: Libraries :: Python Modules',
  'Topic :: Software Development :: Libraries',
]


setup(name = 'diffwave',
    version = VERSION,
    description = DESCRIPTION,
    long_description = open('README.md', 'r').read(),
    long_description_content_type = 'text/markdown',
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    url = URL,
    license = LICENSE,
    keywords = KEYWORDS,
    packages = find_packages('src'),
    package_dir = { '': 'src' },
    install_requires = [
        'numpy',
        'torch>=1.6',
        'torchaudio>=0.6.0',
        'tqdm'
    ],
    classifiers = CLASSIFIERS)
