# -*- coding: utf-8 -*-

# Copyright 2019 Spark-Fuel Authors. All Rights Reserved.
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
from setuptools import setup, find_packages

setup(
    name='pyangel',
    packages=find_packages(exclude=['tests.*', 'tests']),
    version='1.0.0',
    description='angel+ps',
    long_description="long long ago",
    long_description_content_type='text/markdown',
    author='laurie chen',
    url='https://github.com',
    keywords=['parameter server', 'machine learning', 'ai'],
    install_requires=[
        'grpcio>=1.16.1',
        'grpclib>=0.2.5',
        'numpy>=1.16.4',
        'protobuf>=3.8.0',
        'pyarrow>=0.14.1',
        'scipy>=1.2.1'],
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'

    ]
)
