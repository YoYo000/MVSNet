from cnn_wrapper.network import Network

class UniNetDS2(Network):
    """network for 2D image feature extraction definition. Downsampling by 2"""

    def setup(self):
        print ('2D with 32 filters')
        base_filter = 8
        (self.feed('data')
        .conv_bn(3, base_filter, 1, name='conv0_0')
        .conv_bn(3, base_filter, 1, name='conv0_1')
        .conv_bn(5, base_filter * 2, 2, name='conv1_0')
        .conv_bn(3, base_filter * 2, 1, name='conv1_1')
        .conv_bn(3, base_filter * 2, 1, name='conv1_2')
        .conv_bn(5, base_filter * 4, 2, name='conv2_0')
        .conv_bn(3, base_filter * 4, 1, name='conv2_1')
        .conv(3, base_filter * 4, 1, relu = False, name='conv2_2'))

class RegNetUS0(Network):
    """network for regularizing 3D cost volume in a encoder-decoder style. Keeping original size"""

    def setup(self):
        print ('3D with 8 filters')
        base_filter = 8
        (self.feed('data')
        .conv_bn(3, base_filter * 2, 2, name='3dconv1_0')
        .conv_bn(3, base_filter * 4, 2, name='3dconv2_0')
        .conv_bn(3, base_filter * 8, 2, name='3dconv3_0'))

        (self.feed('data')
        .conv_bn(3, base_filter, 1, name='3dconv0_1'))

        (self.feed('3dconv1_0')
        .conv_bn(3, base_filter * 2, 1, name='3dconv1_1'))

        (self.feed('3dconv2_0')
        .conv_bn(3, base_filter * 4, 1, name='3dconv2_1'))

        (self.feed('3dconv3_0')
        .conv_bn(3, base_filter * 8, 1, name='3dconv3_1')
        .deconv_bn(3, base_filter * 4, 2, name='3dconv4_0'))

        (self.feed('3dconv4_0', '3dconv2_1')
        .add(name='3dconv4_1')
        .deconv_bn(3, base_filter * 2, 2, name='3dconv5_0'))

        (self.feed('3dconv5_0', '3dconv1_1')
        .add(name='3dconv5_1')
        .deconv_bn(3, base_filter, 2, name='3dconv6_0'))

        (self.feed('3dconv6_0', '3dconv0_1')
        .add(name='3dconv6_1')
        .conv(3, 1, 1, relu=False, name='3dconv6_2'))

class RefineNet(Network):
    """network for depth map refinement using original image"""

    def setup(self):

        (self.feed('color_image', 'depth_image')
        .concat(axis=3, name='concat_image'))

        (self.feed('concat_image')
        .conv_bn(3, 32, 1, name='refine_conv0')
        .conv_bn(3, 32, 1, name='refine_conv1')
        .conv_bn(3, 32, 1, name='refine_conv2')
        .conv(3, 1, 1, relu=False, name='refine_conv3'))

        (self.feed('refine_conv3', 'depth_image')
        .add(name='refined_depth_image'))
    