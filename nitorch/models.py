import numpy as np
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nitorch.data import show_brain


class _CAE_3D(nn.Module):
    '''
    Parent Convolutional Autoencoder class for 3D images. 
    All other Convolutional Autoencoder classes must inherit from this class.
    '''
    def __init__(self, conv_channels):        
        super().__init__()        
        # check if there are multiple convolution layers within a layer of the network or not
        self.is_nested_conv = ([isinstance(each_c, (list, tuple)) for each_c in conv_channels])
        if(any(self.is_nested_conv) and not all(self.is_nested_conv)):
             raise TypeError(" `conv_channels` can't be a mixture of both lists and ints.")
        self.is_nested_conv = any(self.is_nested_conv)
        
        self.layers = len(conv_channels)
        self.conv_channels = self._format_channels(conv_channels, self.is_nested_conv)


    def _format_channels(self, conv_channels, is_nested_conv = False):
        channels = []
        if(is_nested_conv):
            for i in range(len(conv_channels)):
                    inner_channels = []
                    for j in range(len(conv_channels[i])):
                        if (i == 0) and (j == 0):
                            inner_channels.append([1, conv_channels[i][j]])
                        elif (j == 0) :
                            inner_channels.append([conv_channels[i-1][-1], conv_channels[i][j]])
                        else:
                            inner_channels.append([conv_channels[i][j-1], conv_channels[i][j]])
                    channels.append(inner_channels)
        else:
            for i in range(len(conv_channels)):
                if (i == 0):
                    channels.append([1, conv_channels[i]])
                else:
                    channels.append([conv_channels[i-1], conv_channels[i]])

        return channels


    def assign_parameter(self, parameter, param_name, enable_nested = True):
        ''' Wrapper for parameters of the Autoencoder. 
        Checks if the len and type of the parameter is acceptable.
        If the parameter is just an single value,
        makes its length equal to the number of layers defined in conv_channels
        '''
        if(isinstance(parameter, int)):
            if(self.is_nested_conv and enable_nested):
                return_parameter = [len(inner_list)*[parameter] for inner_list in self.conv_channels]
            else:
                return_parameter = (self.layers * [parameter])

        elif(isinstance(parameter, (list, tuple))):
            if(len(parameter) != self.layers): 
                raise ValueError("The parameter '{}' can either be a single int \
or must be a list of the same length as 'conv_channels'.".format(
        param_name))
        
            if(self.is_nested_conv and enable_nested):
                if(any(
                    [len(c) != len(p) for c, p in zip(self.conv_channels, parameter)]
                    )):
                    raise ValueError("The lengths of the inner lists of the parameter {} \
have to be same as the 'conv_channels'".format(param_name)) 
            # if all length checks pass just return the parameter
            return_parameter = parameter
            
        else: 
            raise TypeError("Parameter {} is neither a int nor a list/tuple".format(
            param_name))

        return return_parameter


    def add_conv_with_Relu(self, inp_channels, out_channels, kernel_size, padding, stride):
        node = nn.Sequential(
            nn.Conv3d(inp_channels, out_channels, kernel_size, padding = padding, stride = stride),
            nn.ReLU(True))        
        return node
        

    def add_deconv_with_Relu(self, inp_channels, out_channels, kernel_size, padding, stride, out_padding):
        node = nn.Sequential(
            nn.ConvTranspose3d(inp_channels, out_channels, kernel_size
                , padding = padding, stride = stride, output_padding=out_padding),
            nn.ReLU(True))        
        return node
        
        
    def add_pool(self, pool_type, kernel_size, padding, stride):
        if(pool_type == "max"):
            node = nn.MaxPool3d(kernel_size, 
                        padding = padding, 
                        stride = stride, 
                        return_indices = True)
        elif(pool_type == "avg"):
            node = nn.AvgPool3d(kernel_size, 
                        padding = padding, 
                        stride = stride)
        else:
            raise TypeError("Invalid value provided for `pool_type`.\
Allowed values are `max`, `avg`.")
            
        return node
        
        
    def add_unpool(self, pool_type, kernel_size, padding, stride):
        if(pool_type == "max"):
            node = nn.MaxUnpool3d(kernel_size, 
                        padding = padding, 
                        stride = stride)
        elif(pool_type == "avg"):
            node = nn.MaxPool3d(kernel_size, 
                        padding = padding, 
                        stride = stride)
        else:
            raise TypeError("Invalid value provided for `pool_type`.\
Allowed values are `max`, `avg`.")
            
        return node
    
        
    def nested_reverse(self, mylist):
        result = []
        for e in mylist:
            if isinstance(e, (list, tuple)):
                result.append(self.nested_reverse(e))
            else:
                result.append(e)
        result.reverse()
        return result

    
    def visualize_feature_maps(self, features):
        features = features.cpu().detach().numpy()
        
        fig = plt.figure()
        num_features = len(features)
        
        for i, f in enumerate(features, 1):            
            # normalize to range [0, 1] first as the values can be very small            
            if((f.max() - f.min()) != 0):
                f = (f - f.min()) / (f.max() - f.min())
            else:
                print("Feature map is all zeros !")
                continue
             
            idxs = np.nonzero(f)
            vals = np.ravel(f[idxs])

            if(len(vals)):
                # calculate the index where the mean value would lie
                mean_idx = np.average(idxs, axis = 1, weights=vals)
                # calculate the angel ratios for each non-zero val            
                angles = (mean_idx.reshape(-1,1) - idxs)
                angles = angles/ (np.max(abs(angles), axis=1).reshape(-1,1))    
            else: # if all values in f are zero, set dummy angle
                angles = [1, 1, 1]

#             print("values = ",vals)
            ax = fig.add_subplot(num_features, 1, i,
                                  projection='3d')
            ax.set_title("Feature-{} in the bottleneck".format(i))
            ax.quiver(*idxs
                      , angles[0]*vals, angles[1]*vals, angles[2]*vals
                     )
            plt.grid()
            
                 
        
        
        
class CAE_3D(_CAE_3D):
    '''
    3D Convolutional Autoencoder model with only convolution layers. Strided convolution
    can be used for undersampling. 
    '''
    def __init__(self
        , conv_channels
        , conv_kernel = 3
        , conv_padding = 1
        , conv_stride = 1
        , deconv_out_padding = None
        , second_fc_decoder = []
        ):
        '''
        Args:
            conv_channels : A list that defines the number of channels of each convolution layer.
            The length of the list defines the number of layers in the encoder. 
            The decoder is automatically constructed as an exact reversal of the encoder architecture.

            conv_kernel (optional): The size of the 3D convolutional kernels to be used. 
            Can either be a list of same length as `conv_channels` or a single int. In the
             former case each value in the list represents the kernel size of that particular
            layer and in the latter case all the layers are built with the same kernel size as 
            specified.

            conv_padding (optional): The amount of zero-paddings to be done along each dimension.
            Format same as conv_kernel.

            conv_stride (optional): The stride of the 3D convolutions.
            Format same as conv_kernel.

            deconv_out_padding (optional): The additional zero-paddings to be done to the output 
            of ConvTranspose / Deconvolutions in the decoder network.
            By default does (stride-1) number of padding.
            Format same as conv_kernel.
            
            second_fc_decoder (optional): By default this is disabled. 
            If a non-empty list of ints is provided then a secondary fully-connected decoder 
            network is constructed as per the list.
            Each value represents the number of cells in each layer. Just like `conv_channels`
            the length of the list defines the number of layers.
            If enabled, the forward() method returns a list of 2 outputs, one from the Autoencoder's
            decoder and the other from this fully-connected decoder network.            
        '''

        super().__init__(conv_channels)

        assert not(self.is_nested_conv), "The conv_channels must be a list of ints (i.e. number of channels).\
It cannot be a list of lists."

        self.conv_kernel = self.assign_parameter(conv_kernel, "conv_kernel")
        self.conv_padding = self.assign_parameter(conv_padding, "conv_kernel")
        self.conv_stride = self.assign_parameter(conv_stride, "conv_stride")
        if(deconv_out_padding == None):
            deconv_out_padding = [s-1 for s in self.conv_stride]
        self.deconv_out_padding = self.assign_parameter(deconv_out_padding, "deconv_out_padding")
        
        if(second_fc_decoder):
            self.second_fc_decoder = self._format_channels(second_fc_decoder)[1:]
        else:
            self.second_fc_decoder = []

        self.convs = nn.ModuleList()
        self.deconvs = nn.ModuleList()
        
        for i in range(self.layers):
            # build the encoder
            self.convs.append(
                self.add_conv_with_Relu(
                    self.conv_channels[i][0], self.conv_channels[i][1], 
                    self.conv_kernel[i]
                    , self.conv_padding[i]
                    , self.conv_stride[i]
                    )
                )
            # build the decoder
            self.deconvs.append(
                self.add_deconv_with_Relu(
                    self.conv_channels[-i-1][1], self.conv_channels[-i-1][0], 
                    self.conv_kernel[-i-1]
                    , self.conv_padding[-i-1]
                    , self.conv_stride[-i-1]
                    , self.deconv_out_padding[-i-1]
                    )
                )
        if(self.second_fc_decoder):
        # build the second fc decoder
            self.fcs = nn.ModuleList()
            for layer in self.second_fc_decoder:
                self.fcs.append(
                    nn.Linear(layer[0], layer[1])
                )
        

    def forward(self, x, debug = False, visualize_training = False):

            if(debug):
                print("\nImage dims ="+str(x.size()))
                # show only the first image in the batch
            if(visualize_training):
                # show only the first image in the batch
                show_brain(x[0].squeeze().cpu().detach().numpy(), draw_cross = False)
                plt.suptitle("Input Image")
                plt.show()

            #encoder
            for i, conv in enumerate(self.convs):
                x = conv(x)
                if(debug): print("conv{} output dim = {}".format(i+1, x.size()))

            if(self.second_fc_decoder):
                #save the encoder output as a flat array
                encoder_out = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
                
            if(debug):
                print("\nEncoder output dims ="+str(x.size())+"\n")
            if(visualize_training):
                # show only the first image in the batch
                self.visualize_feature_maps(x[0])
                plt.suptitle("Encoder output")
                plt.show()
            
            #decoder
            for i, deconv in enumerate(self.deconvs):
                x = deconv(x)
                if(debug): print("deconv{} output dim = {}".format(i+1, x.size()))
                        
            if(debug):
                print("\nDecoder output dims ="+str(x.size())+"\n")
            if(visualize_training):
                show_brain(x[0].squeeze().cpu().detach().numpy(),  draw_cross = False)
                plt.suptitle("Reconstructed Image")
                plt.show()
                
            if(self.second_fc_decoder):
                # run the second fully connected decoder network
                x2 = self.fcs[0](encoder_out)
                for fc in self.fcs[1:]:
                    x2 = fc(x2)
                if(debug):
                    print("FC decoder output dims ="+str(x2.size()))
                if(visualize_recon):
                    print("FC Output =",x2[0].squeeze().cpu().detach().numpy())
                    
                return [x, x2]
            
            else:
                return x

            
            
class CAE_3D_with_pooling(_CAE_3D):
    '''
    3D Convolutional Autoencoder model with alternating Pooling layers. 
    '''
    def __init__(self
        , conv_channels
        , conv_kernel = 3, conv_padding = 1, conv_stride = 1
        , pool_type = "max"
        , pool_kernel = 2, pool_padding = 0, pool_stride = 2
        , deconv_out_padding = None
        , second_fc_decoder = []
        ):
        '''
        Args:
            conv_channels : A nested list whose length defines the number of layers. Each layer
            can intern have multiple convolutions followed by a layer of Pooling. The lengths of the 
            inner list defines the number of convolutions per such layer and the value defines the number of
            channels for each of these convolutions.
            The decoder is constructed to be simply an exact reversal of the encoder architecture.

            conv_kernel (optional): The size of the 3D convolutional kernels to be used. 
            Can either be a list of lists of same lengths as `conv_channels` or a single int. In the
             former case each value in the list represents the kernel size of that particular
            layer and in the latter case all the layers are built with the same kernel size as 
            specified.

            conv_padding (optional): The amount of zero-paddings to be done along each dimension.
            Format same as conv_kernel.

            conv_stride (optional): The stride of the 3D convolutions.
            Format same as conv_kernel.

            deconv_out_padding (optional): The additional zero-paddings to be done to the output 
            of ConvTranspose / Deconvolutions in the decoder network.
            By default does (stride-1) number of padding.
            Format same as conv_kernel.
            
            pool_type (optional): The type of pooling to be used. Options are (1)"max"  (2)"avg" 
            
            pool_kernel, pool_padding, pool_stride (optional): Can either be a single int or a list
            of respective pooling parameter values.
            The length of these list must be same as length of conv_channels i.e. the number of layers. 
            
            second_fc_decoder (optional): By default this is disabled. 
            If a non-empty list of ints is provided then a secondary decoder of a fully-connected network  
            is constructed as per the list.
            Each value represents the number of neurons in each layer. The length of the list
            defines the number of layers.  
            It has to be taken care to ensure that the number of parameters of the output of the encoder 
            is equal to the number of neurons in the 1st layer of this fc-decoder network.
            
            If enabled, the forward() method returns a list of 2 outputs, one from the Autoencoder's
            decoder and the other from this fully-connected decoder.
        '''
        
        super().__init__(conv_channels)
        
        assert (self.is_nested_conv), "The conv_channels must be a list of list of ints Ex. [[16],[32 64],[64],...] (i.e. number of channels).\
It cannot be a list."
        
        self.conv_kernel = self.assign_parameter(conv_kernel, "conv_kernel")
        self.conv_padding = self.assign_parameter(conv_padding, "conv_padding")
        self.conv_stride = self.assign_parameter(conv_stride, "conv_stride")
        self.pool_kernel = self.assign_parameter(pool_kernel, "pool_kernel", enable_nested=False)
        self.pool_padding = self.assign_parameter(pool_padding, "pool_padding", enable_nested=False)
        self.pool_stride = self.assign_parameter(pool_stride, "pool_stride", enable_nested=False)
        self.reversed_conv_channels = self.nested_reverse(self.conv_channels)
        self.reversed_conv_kernel = self.nested_reverse(self.conv_kernel)
        self.reversed_conv_padding = self.nested_reverse(self.conv_padding)
        self.reversed_conv_stride = self.nested_reverse(self.conv_stride)
        
        if(deconv_out_padding == None):
            self.deconv_out_padding = [[s-1 for s in layer] for layer in self.reversed_conv_stride]
        else:
            self.deconv_out_padding = self.nested_reverse(
                self.assign_parameter(deconv_out_padding, "deconv_out_padding")
            )
        
        if(second_fc_decoder):
            self.second_fc_decoder = self._format_channels(second_fc_decoder)[1:]
        else:
            self.second_fc_decoder = []
            
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.deconvs = nn.ModuleList()
        self.unpools = nn.ModuleList()
        

        for i in range(self.layers):
            
            self.convs.append(
                nn.ModuleList(
                    [self.add_conv_with_Relu(
                        inner_conv_channels[0], inner_conv_channels[1]
                        , self.conv_kernel[i][j]
                        , self.conv_padding[i][j]
                        , self.conv_stride[i][j]) \
                    for j, inner_conv_channels in enumerate(self.conv_channels[i])]
                    )
                )

            self.deconvs.append(
                nn.ModuleList(
                    [self.add_deconv_with_Relu(
                        inner_deconv_channels[0], inner_deconv_channels[1] 
                        , self.reversed_conv_kernel[i][j]
                        , self.reversed_conv_padding[i][j]
                        , self.reversed_conv_stride[i][j]
                        , self.deconv_out_padding[i][j]) \
                    for j, inner_deconv_channels in enumerate(self.reversed_conv_channels[i])]
                    )
                )

            self.pools.append(
                self.add_pool(
                    pool_type,
                    self.pool_kernel[i], 
                    stride = self.pool_stride[i], 
                    padding = self.pool_padding[i]
                )
            )
            self.unpools.append(
                self.add_unpool(
                    pool_type,
                    self.pool_kernel[-i-1], 
                    stride = self.pool_stride[-i-1], 
                    padding = self.pool_padding[-i-1]
                )
            )
        
        if(self.second_fc_decoder):
            # build the second fc decoder
            self.fcs = nn.ModuleList()
            for layer in self.second_fc_decoder:
                self.fcs.append(
                    nn.Linear(layer[0], layer[1])
                )
        

    def forward(self, x, debug=False, visualize_training=False):
            pool_idxs = []
            pool_sizes = [x.size()] #https://github.com/pytorch/pytorch/issues/580
            
            if(debug):
                print("\nImage dims ="+str(x.size()))
            if(visualize_training):
                # show only the first image in the batch
                show_brain(x[0].squeeze().cpu().detach().numpy(),  draw_cross = False)
                plt.suptitle("Input image")
                plt.show()

            #encoder
            for i,(convs, pool) in enumerate(zip(self.convs, self.pools)):
                for j, conv in enumerate(convs):
                    x = conv(x)
                    if(debug):print("conv{}{} output dim = {}".format(i+1, j+1, x.size()))
                        
                x, idx = pool(x)
                pool_sizes.append(x.size()) 
                pool_idxs.append(idx)
                if(debug):print("pool{} output dim = {}".format(i+1, x.size()))

            if(debug):
                print("\nEncoder output dims ="+str(x.size())+"\n")
            if(visualize_training):                
                self.visualize_feature_maps(x[0]) # show only the first image in the batch
                plt.suptitle("Encoder output")
                plt.show()
            
            if(self.second_fc_decoder):
                #save the encoder output as a flat array
                encoder_out = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
                
            #decoder
            pool_sizes.pop() # pop out the last size as it is not necessary

            for i,(deconvs, unpool) in enumerate(zip(self.deconvs, self.unpools)):

                x = unpool(x, pool_idxs.pop(), output_size=pool_sizes.pop())
                if(debug):print("unpool{} output dim = {}".format(i+1, x.size()))
                
                for j, deconv in enumerate(deconvs):
                    x = deconv(x)
                    if(debug):print("deconv{}{} output dim = {}".format(i+1, j+1, x.size()))
                        
            if(debug):
                print("\nDecoder output dims ="+str(x.size())+"\n")
            if(visualize_training):
                show_brain(x[0].squeeze().cpu().detach().numpy(),  draw_cross = False)
                plt.suptitle("Reconstructed Image")
                plt.show()
                
            if(self.second_fc_decoder):
                # run the second fully connected decoder network
                x2 = self.fcs[0](encoder_out)
                for fc in self.fcs[1:]:
                    x2 = fc(x2)
                if(debug):
                    print("FC decoder output dims ="+str(x2.size()))
                if(visualize_training):
                    print("\nFC Output =",x2[0].squeeze().cpu().detach().numpy())
                    
                return [x, x2]
            
            else:
                return x
            