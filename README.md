# nilmtk-dl
基于深度学习的非侵入式负荷检测工具包。在nilmtk-contrib的基础上做了一些改进。主要有：
1. 修改了一些bug（当然也可能是因为我不会用）；
2. 增加一些metrics，现在不仅可以用Energy-based标准评估，也可以用Event-based标准进行评估；
3. 增加了一个激活转换功能，在数据集仅提供功率时，也能够通过激活函数将功率转化为启停事件，方便一些以启停事件作为目标函数的模型复现（不过我复现后觉得那些模型效果不好）；
4. 由于REDD数据集存在一些bad section，增加了自动提取nilmtk.DataSet中的good section进行训练与预测的功能；
5. 增加了一些可视化功能；
6. 对原有的基于深度学习Disaggregator中的保存模型、读取模型函数进行补充，现在可以通过简单更改实验配置文件就能够实现模型的保存或读取。

ex_time_\*.py与ex_house_\*.py是针对REDD数据集的实验配置文件范例，\*为电器简写。

两个jupyter notebook是草稿本

由于本人水平十分有限，代码可能有一些bug，有任何问题请联系xieck13@gmail.com
