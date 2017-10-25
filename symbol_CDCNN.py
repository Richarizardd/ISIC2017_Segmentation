import mxnet as mx
import numpy as np

mx.random.seed(1301)

def print_inferred_shape(name, net):
    ar, ou, au = net.infer_shape(data=(1, 3, 256, 256))
    print name+":", ou

def residual_unit(data, num_filter, name, short_skip=False, bn_mom=0.9, workspace=256):
    conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=None, workspace=256, no_bias=True, name=name+'_conv1')
    print_inferred_shape(name+'_conv1', conv1)
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name+'_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name+'_relu1')
    
    conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), workspace=256, no_bias=True, name=name+'_conv2')
    print_inferred_shape(name+'_conv2', conv2)
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name+'_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name+'_relu2')
    conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), workspace=256, no_bias=True, name=name+'_conv3')
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name+'_bn3')

    if short_skip:
        eltwise = conv1+conv3
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name+'_relu3')
    else:
        return mx.sym.Activation(data=conv3, act_type='relu', name=name+'_relu3')

def residual_unit2(data, num_filter, name, short_skip=False, bn_mom=0.9, workspace=256):
    conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=None, workspace=256, no_bias=True, name=name+'_conv1')
    print_inferred_shape(name+'_conv1', conv1)
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name+'_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name+'_relu1')
    
    conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), workspace=256, no_bias=True, name=name+'_conv2')
    print_inferred_shape(name+'_conv2', conv2)
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name+'_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name+'_relu2')
    conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), workspace=256, no_bias=True, name=name+'_conv3')

    if short_skip:
        eltwise = conv1+conv3
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name+'_relu3')
    else:
        return mx.sym.Activation(data=conv3, act_type='relu', name=name+'_relu3')




def CDCNN(short_skip=False, long_skip="Concat"):
    data = mx.symbol.Variable("data")
    print_inferred_shape("Stage1_Input", data)
    stage1d = residual_unit(data, 32, "Stage1D"); print
    
    stage2d = mx.symbol.Pooling(data=stage1d, name='Stage2D_pool', pool_type="max", kernel=(2, 2), stride=(2, 2))
    print_inferred_shape("Stage2D_pool", stage2d)
    stage2d = residual_unit(stage2d, 64, "Stage2D", short_skip); print
    
    stage3d = mx.symbol.Pooling(data=stage2d, name='Stage3D_pool', pool_type="max", kernel=(2, 2), stride=(2, 2))
    print_inferred_shape("Stage3D_pool", stage3d)
    stage3d = residual_unit(stage3d, 128, "Stage3D", short_skip); print
    
    stage4d = mx.symbol.Pooling(data=stage3d, name='Stage4D_pool', pool_type="max", kernel=(2, 2), stride=(2, 2))
    stage4d = mx.sym.Dropout(stage4d, p=0.25)
    print_inferred_shape("Stage4D_pool", stage4d)
    stage4d = residual_unit(stage4d, 256, "Stage4D", short_skip); print
    
    stage5 = mx.symbol.Pooling(data=stage4d, name='Stage5_pool', pool_type="max", kernel=(2, 2), stride=(2, 2))
    print_inferred_shape("Stage5_pool", stage5); print
    stage5 = residual_unit(stage5, 512, "Stage5", short_skip); print
    
    stage4u = mx.symbol.Deconvolution(stage5, name='Stage4U_deconv', kernel=(2, 2), stride=(2, 2), num_filter=256)
    print_inferred_shape("Stage4U_deconv", stage4u)
    if long_skip:
        stage4u = mx.sym.Concat(*[stage4d, stage4u])
        print_inferred_shape("Stage4U_concat", stage4u)
    stage4u = mx.sym.Dropout(stage4u, p=0.25)
    stage4u = residual_unit(stage4u, 256, "Stage4U", short_skip); print
    
    stage3u = mx.symbol.Deconvolution(stage4u, name='Stage3U_deconv', kernel=(2, 2), stride=(2, 2), num_filter=128)
    print_inferred_shape("Stage3U_deconv", stage3u)
    if long_skip == "Add":
        stage3u = stage3d + stage3u
        print_inferred_shape("Stage3U_concat", stage3u)
    elif long_skip == "Concat":
        stage3u = mx.sym.Concat(*[stage3d, stage3u])
        print_inferred_shape("Stage3U_concat", stage3u)
    stage3u = residual_unit(stage3u, 128, "Stage3U", short_skip); print
    
    stage2u = mx.symbol.Deconvolution(stage3u, name='Stage2U_deconv', kernel=(2, 2), stride=(2, 2), num_filter=64)
    print_inferred_shape("Stage2U_deconv", stage2u)
    if long_skip == "Add":
        stage2u = stage2d + stage2u
        print_inferred_shape("Stage2U_concat", stage2u)
    elif long_skip == "Concat":
        stage2u = mx.sym.Concat(*[stage2d, stage2u])
        print_inferred_shape("Stage2U_concat", stage2u)
    stage2u = residual_unit(stage2u, 64, "Stage2U", short_skip); print
    
    stage1u = mx.symbol.Deconvolution(stage2u, name='Stage1U_deconv', kernel=(2, 2), stride=(2, 2), num_filter=32)
    print_inferred_shape("Stage1U_deconv", stage1u)
    if long_skip == "Add":
        stage1u = stage1d + stage1u
        print_inferred_shape("Stage1U_concat", stage1u)
    elif long_skip == "Concat":
        stage1u = mx.sym.Concat(*[stage1d, stage1u])
        print_inferred_shape("Stage1U_concat", stage1u)
    stage1u = residual_unit(stage1u, 32, "Stage1U", short_skip); print
    
    stage1_output = mx.symbol.Convolution(name='Stage1_conv_output', data=stage1u, kernel=(1, 1), stride=(1, 1), num_filter=1)
    print_inferred_shape("Stage1_output", stage1_output)
    stage1_output = mx.symbol.Activation(name='Stage1_relu_output', data=stage1_output, act_type="relu")
    stage1_output = mx.symbol.Flatten(stage1_output)
    return mx.symbol.LogisticRegressionOutput(data=stage1_output, name='softmax')
