from cmath import log
from data import MatrixDataset
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random
from utils import logger
from model import UPCCModel

"""
RESULT UPCC(PCC):
Density:0.05,type:rt,mae:0.6989300707521433,mse:2.7737428845756824,rmse:1.6654557588166916
Density:0.1,type:rt,mae:0.5598082846452833,mse:2.1501265216568024,rmse:1.4663309727536966
Density:0.15,type:rt,mae:0.49665146697083623,mse:1.822202449824236,rmse:1.349889791732731
Density:0.2,type:rt,mae:0.4640162396499109,mse:1.6248480178887317,rmse:1.2746952647157406

Density:0.05,type:tp,mae:36.44804703916447,mse:7770.4758688545135,rmse:88.15030271561473
Density:0.1,type:tp,mae:24.80577811244719,mse:4828.352330817265,rmse:69.4863463625572
Density:0.15,type:tp,mae:20.103870591282085,mse:3577.5086548164077,rmse:59.812278462004834
Density:0.2,type:tp,mae:17.75683423278253,mse:2891.0985679440246,rmse:53.76893683107399



Density:0.05,type:tp,mae:31.433279057978826,mse:5942.681984715335,rmse:77.08879286067031
Density:0.1,type:tp,mae:24.705758868958824,mse:4119.820902493704,rmse:64.18583101038503
Density:0.15,type:tp,mae:22.348603068910084,mse:3475.7766495526553,rmse:58.95571770025919
Density:0.2,type:tp,mae:21.212877869416783,mse:3153.9556887043786,rmse:56.16008982101416

------------------

RESULT UPCC(COS):
Density:0.05,type:rt,mae:0.8816043906403586,mse:3.4499158310187976,rmse:1.8573949044343794
Density:0.1,type:rt,mae:0.877630165969834,mse:3.444233792894344,rmse:1.8558647022060482
Density:0.15,type:rt,mae:0.8743843800785683,mse:3.444289466204811,rmse:1.8558797014367099
Density:0.2,type:rt,mae:0.873491906493263,mse:3.4519766422363864,rmse:1.8579495801114696

------------------

RESULT UPCC(ACOS):
Density:0.05,type:rt,mae:0.754261004589736,mse:2.955304424033613,rmse:1.7190998877417254
Density:0.1,type:rt,mae:0.6226659481102241,mse:2.340810418516644,rmse:1.5299707247253602
Density:0.15,type:rt,mae:0.5344715968678315,mse:1.9297844656565475,rmse:1.3891668242714938
Density:0.2,type:rt,mae:0.4941239153511182,mse:1.7295844645305374,rmse:1.31513667142641
"""

freeze_random()  # 冻结随机数 保证结果一致

for density in [0.05, 0.1, 0.15, 0.2]:
    type_ = "rt"
    rt_data = MatrixDataset(type_)
    train_data, test_data = rt_data.split_train_test(density)

    umean = UPCCModel()
    umean.fit(train_data, metric='PCC')
    y, y_pred = umean.predict(test_data, 20)

    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")
    logger.info(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")
