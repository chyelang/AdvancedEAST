# AdvancedEAST
modified by chyelang

AdvancedEAST is an algorithm used for Scene image text detect,
which is primarily based on
[EAST:An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2),
and the significant improvement was also made,
which make long text predictions more accurate.
If this project is helpful to you, welcome to star.
And if you have any problem, please contact me.
* email:yijie.huo@foxmail.com
* wechat:gekongdianxue

# advantages
* writen in keras, easy to read and run
* base on EAST, an advanced text detect algorithm
* easy to train the model
* significant improvement was made, long text predictions more accurate.(please
see 'demo results' part bellow,
and pay attention to the activation image,
which starts with yellow grids, and ends with green grids.) 

In my experiments,
AdvancedEast has obtained much better prediction accuracy then East,
especially on long text. Since East calculates final vertexes coordinates with
weighted mean values of predicted vertexes coordinates of all pixels. It is too
difficult to predict the 2 vertexes from the other side of the quadrangle.

# project files
* config file:cfg_local.py and cfg_server.py, control parameters.
you need to use --section server argument to use cfg_server.py, or cfg_local is used by default.
如果你还有别的cfg文件，你可以将其命名为别的名字，要用时再为其重命名为cfg_server.py
* pre-process data:
    preprocess.py,resize image
* label data:
    label.py,produce label info
* define network
    network.py
* define loss function
    losses.py
* execute training
    advanced_east.py and data_generator.py
* predict
    predict.py and nms.py

# network arch
* AdvancedEast

![AdvancedEast network arch](image/AdvancedEast.network.png "AdvancedEast network arch")

* East

![East network arch](image/East.network.png "East network arch")


# setup
* python 3.5.0+
* tensorflow-gpu 1.4.0+(or tensorflow 1.4.0+)
* keras 2.1.4+
* numpy 1.14.1+
* tqdm 4.19.7+
* h5py
* tqdm

# training
* prepare training data:make data root dir(icpr),
copy images to root dir, and copy txts to root dir,
data format details could refer to 'ICPR MTWI 2018 挑战赛二：网络图像的文本检测',
[Link](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.3bcad780oQ9Ce4&raceId=231651)
* modify config params in cfg_local.py or config_server.py, depending on the machine you are running your codes, see default values.
you can specify --section local/server for the following commands, or the default --section local will be used.
if you want to specify a gpu to use, you need to modify Line 3 in advanced_east.py
* python preprocess.py, resize image to 256*256,384*384,512*512,640*640,736*736,
and train respectively could speed up training process. 每次预测只能选择一个图片尺寸？
* python label.py
* python advanced_east.py
* python predict.py to predict. You may need to specify a valid image path in main function of predict.py. 该函数可以在一张图上使用你的模型做预测，
给出两张图：一张是文字segmentation，另一张是用四边形框起来的文字。
* python tianchi_submit.py to get submission file for test sets. 你需要先把icpr_mtwi_task2/image_test拷贝到cfg.data_dir之下
* python tianchi_check.py to verify the prediction performance. you might need to read the code to prepare some data

# License
The codes are released under the MIT License.

# references
* [EAST:An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2)

* [CTPN:Detecting Text in Natural Image with Connectionist Text Proposal Network](https://arxiv.org/abs/1609.03605)

