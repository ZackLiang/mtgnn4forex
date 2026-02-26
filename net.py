from layer import *


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device,
                 predefined_A=None, static_feat=None, dropout=0.3,
                 subgraph_size=20, node_dim=40, dilation_exponential=1,
                 conv_channels=32, residual_channels=32, skip_channels=64,
                 end_channels=128, seq_length=12, in_dim=2, out_dim=12,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True,
                 revin=True, dual_graph=True, use_router=True):
        super(gtnet, self).__init__()
        self.gcn_true    = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes   = num_nodes
        self.dropout     = dropout
        self.predefined_A = predefined_A

        self.filter_convs    = nn.ModuleList()
        self.gate_convs      = nn.ModuleList()
        self.residual_convs  = nn.ModuleList()
        self.skip_convs      = nn.ModuleList()
        self.gconv1          = nn.ModuleList()   # Expert A：动态自适应图 GCN
        self.gconv2          = nn.ModuleList()
        self.norm            = nn.ModuleList()

        # Expert B：格兰杰先验图 GCN（仅 use_router=True 时启用）
        self.expert_gconv1   = nn.ModuleList()
        self.expert_gconv2   = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim,
                                    device, alpha=tanhalpha,
                                    static_feat=static_feat)
        self.seq_length = seq_length

        # ── 创新点 1: RevIN ──────────────────────────────────────────
        self.revin_enabled = revin
        if self.revin_enabled:
            self.revin = RevIN(num_nodes, affine=True)

        # ── 创新点 3: RegimeMoE Router ───────────────────────────────
        self.use_router = use_router
        if self.use_router:
            # 轻量感知机：输入当前窗口波动率 stdev → 输出 alpha ∈ [0,1]
            # alpha = 1 → 完全用动态图专家(Expert A)
            # alpha = 0 → 完全用格兰杰图专家(Expert B)
            self.router = nn.Sequential(
                nn.Conv2d(in_dim, 16, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, 1),
            )
            # 初始 alpha ≈ 0.5（两专家等权），温度=0.5 下偏置=-0
            # sigmoid(0/0.5) = 0.5，让两路专家从均衡起点各自分化
            with torch.no_grad():
                self.router[-1].bias.fill_(0.0)

        # ── 创新点 2: Dual Graph（固定融合权重，M2 专用）───────────────
        self.dual_graph = dual_graph
        if self.dual_graph:
            # 初始值 0.1，训练中自适应调整
            self.fusion_weight = nn.Parameter(
                torch.tensor(0.1, dtype=torch.float32), requires_grad=True)
        if self.dual_graph and self.predefined_A is None:
            print("Warning: Dual Graph is enabled but predefined_A is None!")

        # ── 感受野 & 各层模块 ─────────────────────────────────────────
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1)
                    * (dilation_exponential ** layers - 1)
                    / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1

            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1)
                        * (dilation_exponential ** j - 1)
                        / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels,
                                      dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels,
                                      dilation_factor=new_dilation))
                self.residual_convs.append(
                    nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1)))

                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(conv_channels, skip_channels,
                                  kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(
                        nn.Conv2d(conv_channels, skip_channels,
                                  kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    # Expert A（动态图）
                    self.gconv1.append(
                        mixprop(conv_channels, residual_channels,
                                gcn_depth, dropout, propalpha))
                    self.gconv2.append(
                        mixprop(conv_channels, residual_channels,
                                gcn_depth, dropout, propalpha))
                    # Expert B（格兰杰图，仅 use_router=True 时实际使用）
                    if self.use_router:
                        self.expert_gconv1.append(
                            mixprop(conv_channels, residual_channels,
                                    gcn_depth, dropout, propalpha))
                        self.expert_gconv2.append(
                            mixprop(conv_channels, residual_channels,
                                    gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm((residual_channels, num_nodes,
                                   self.seq_length - rf_size_j + 1),
                                  elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(
                        LayerNorm((residual_channels, num_nodes,
                                   self.receptive_field - rf_size_j + 1),
                                  elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels,
                                    kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim,
                                    kernel_size=(1, 1), bias=True)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_dim, skip_channels,
                                   kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1),
                                   bias=True)
        else:
            self.skip0 = nn.Conv2d(in_dim, skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels,
                                   kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)

    # ──────────────────────────────────────────────────────────────────
    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, \
            'input sequence length not equal to preset sequence length'

        # ── RevIN 归一化 ─────────────────────────────────────────────
        if self.revin_enabled:
            input, stdev = self.revin(input, 'norm')
        else:
            stdev = None

        # ── RegimeMoE Router：计算体制权重 alpha ─────────────────────
        if self.use_router and stdev is not None:
            raw_alpha = self.router(stdev)
            # temperature=0.5：平滑变化，避免梯度消失
            alpha = torch.sigmoid(raw_alpha / 0.5)
            self.router_alpha_tensor = alpha          # 保留计算图供正则化
            self.last_alpha = alpha.detach().cpu().numpy()
            alpha = alpha.view(-1, 1, 1, 1)
        else:
            alpha = None
            self.router_alpha_tensor = None

        # ── Padding ──────────────────────────────────────────────────
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(
                input, (self.receptive_field - self.seq_length, 0, 0, 0))

        # ── 动态自适应图 ──────────────────────────────────────────────
        adp_learned = None
        if self.gcn_true:
            if self.buildA_true:
                adp_learned = self.gc(self.idx if idx is None else idx)
                self.last_adp = adp_learned.detach().cpu().numpy()
            else:
                adp_learned = self.predefined_A

        # ── 主干网络 ──────────────────────────────────────────────────
        x    = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter_  = torch.tanh(self.filter_convs[i](x))
            gate_    = torch.sigmoid(self.gate_convs[i](x))
            x = filter_ * gate_
            x = F.dropout(x, self.dropout, training=self.training)
            skip = self.skip_convs[i](x) + skip

            if self.gcn_true:
                # Expert A：动态自适应图
                x_A = (self.gconv1[i](x, adp_learned)
                       + self.gconv2[i](x, adp_learned.transpose(1, 0)))

                if self.use_router and alpha is not None \
                        and self.predefined_A is not None:
                    # ── 创新点 3: RegimeMoE 双专家凸组合 ────────────
                    # Expert B：格兰杰先验图（独立权重，专门学习有向因果传播）
                    x_B = (self.expert_gconv1[i](x, self.predefined_A)
                           + self.expert_gconv2[i](x, self.predefined_A.transpose(1, 0)))
                    # alpha 由 Router 根据波动率动态决定：
                    #   高波动 → alpha → 1 → 更信任动态图（实时捕捉市场联动）
                    #   低波动 → alpha → 0 → 更信任格兰杰图（因果结构更稳定）
                    x = alpha * x_A + (1.0 - alpha) * x_B

                elif self.dual_graph and self.predefined_A is not None:
                    # ── 创新点 2: M2 固定加法融合（无路由）─────────────
                    x_static = (self.gconv1[i](x, self.predefined_A)
                                + self.gconv2[i](x, self.predefined_A.transpose(1, 0)))
                    x = x_A + self.fusion_weight * x_static
                else:
                    x = x_A
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx if idx is None else idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # ── RevIN 反归一化 ────────────────────────────────────────────
        if self.revin_enabled:
            x = self.revin(x, 'denorm', target_idx=0)
        return x
