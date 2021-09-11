import matplotlib.pyplot as plt


class Layers:
    def maxPooling2D(self, x):
        return {'class_desc': f"{x['name']} pool_size={x['pool_size']} strides={x['strides']}", 'param': False}

    def conv2D(self, x):
        return {'name': x['name'], 'size': x['filters'], 'kernel_size': x['kernel_size'], 'activation': x['activation'], 'param': True}

    def flatten(self, x):
        return {'class_desc': x['name'], 'param': False}

    def inputLayer(self, x):
        return {'class_desc': x['name'], 'param': False}

    def dense(self, x):

        return {'name': x['name'], 'size': x['units'], 'activation': x['activation'], 'param': True}

    def sequential(self, x):
        return {'class_desc': x['name'], 'param': False}

    def activation(self, x):
        return {'class_desc': x['name'], 'activation': x['activation'], 'param': False}

    def dropout(self, x):
        if x['noise_shape']:
            return {'class_desc': f"{x['name']} rate: {x['rate']} noise_shape: {x['noise_shape']}", 'param': False}

        return {'class_desc': f"{x['name']} rate: {x['rate']}", 'param': False}


class DrawNN(Layers):
    def __init__(self, model, layout='h', dpi=70, sparse=1, marker='*', grid=False,
                 color_layer_param='green', color_layer_no_param='red', title=True,
                 legend=True, textsize=16, n_p_s=2.0, name='name', format='svg', save=False):

        self.layout = layout
        self.dpi = dpi
        self.model = model
        self.sparse = sparse
        self.title = title
        self.grid = grid
        self.legend = legend
        self.textsize = textsize
        self.marker = marker
        self.color_layer_param = color_layer_param
        self.color_layer_no_param = color_layer_no_param
        self.n_p_s = n_p_s
        self.name = name
        self.format = format
        self.save = save

        super().__init__()

        self.layer_class_name = {'MaxPooling2D': Layers.maxPooling2D, 'InputLayer': Layers.inputLayer, 'Conv2D': Layers.conv2D,
                                 'Flatten': Layers.flatten, 'Sequential': Layers.sequential, 'Dense': Layers.dense, 'Activation': Layers.activation, 'Dropout': Layers.dropout}

    def create_graph(self):

        all_layers = self.model.get_config()['layers']
        plt.figure(figsize=(20, 16), dpi=self.dpi)
        if not self.grid:
            plt.axis('off')
            plt.grid(b=None)

        list_of_makers = [self.layer_class_name[all_layers[i]['class_name']](self, all_layers[i]['config'])['size'] for i in range(
            len(all_layers)) if self.layer_class_name[all_layers[i]['class_name']](self, all_layers[i]['config'])['param']]

        for i in range(len(all_layers)):
            config_of_a_layer = self.layer_class_name[all_layers[i]['class_name']](
                self, all_layers[i]['config'])
            if self.layout == 'h':
                xcor_max = max(list_of_makers)
                ycor_max = len(all_layers)
                plt.ylim(0, ycor_max+.5)
                if config_of_a_layer['param']:
                    no_of_points = config_of_a_layer['size']
                    for j in range(1, no_of_points+1, self.sparse):

                        plt.scatter(xcor_max/2-no_of_points//2+j,
                                    i+1, marker=self.marker, linewidths=self.n_p_s)
                    text = f"{no_of_points} units  {config_of_a_layer['name']}  {config_of_a_layer['activation']}"
                    plt.text(xcor_max//2, i+.55, text, color=self.color_layer_param,
                             horizontalalignment='center', fontsize=self.textsize)
                else:
                    if 'activation' in config_of_a_layer:
                        text = f"{config_of_a_layer['class_desc']} {config_of_a_layer['activation']}"
                    else:
                        text = str(config_of_a_layer['class_desc'])
                    plt.text(xcor_max//2, i+.55, text, color=self.color_layer_no_param,
                             horizontalalignment='center', fontsize=self.textsize)

            if self.layout == 'v':
                xcor_max = len(all_layers)
                ycor_max = max(list_of_makers)
                plt.xlim(0, len(all_layers)+.5)

                if config_of_a_layer['param']:
                    no_of_points = config_of_a_layer['size']
                    for j in range(1, no_of_points+1, self.sparse):
                        plt.scatter(i+1, ycor_max/2-no_of_points//2+j,
                                    marker=self.marker, linewidths=self.n_p_s)
                    text = f"{no_of_points} units  {config_of_a_layer['name']}  {config_of_a_layer['activation']}"
                    plt.text(i+.55, ycor_max//2, text, color=self.color_layer_param,
                             verticalalignment='center', fontsize=self.textsize, rotation=90)
                else:
                    if 'activation' in config_of_a_layer:
                        text = f"{config_of_a_layer['class_desc']} {config_of_a_layer['activation']}"
                    else:
                        text = str(config_of_a_layer['class_desc'])
                    plt.text(i+.55, ycor_max//2, text, color=self.color_layer_no_param,
                             verticalalignment='center', fontsize=self.textsize, rotation=90)

        if self.legend:
            plt.text(xcor_max, ycor_max, f'Layers with parameters({self.color_layer_param})\n\nLayers with no params({self.color_layer_no_param})',
                     ha='right', backgroundcolor='#EEE7E6', fontsize=self.textsize)
        if self.title:
            plt.title(
                f'Total trainable parameters(weights + biases)= {self.model.count_params()}', fontsize=28, fontfamily='cursive')

        plt.tight_layout()
        if self.save:
            self.save_plot()
        plt.show()

    def save_plot(self):
        plt.savefig(f"{self.name}.{self.format}", format=self.format)
