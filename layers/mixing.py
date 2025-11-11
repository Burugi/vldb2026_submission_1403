import torch
import torch.nn as nn

class SeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """
    def __init__(self, seq_len: int, d_model: int, down_sampling_layers: int = 2, down_sampling_window: int = 2):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    seq_len // (down_sampling_window ** i),
                    seq_len // (down_sampling_window ** (i + 1))
                ),
                nn.GELU(),
                nn.Linear(
                    seq_len // (down_sampling_window ** (i + 1)),
                    seq_len // (down_sampling_window ** (i + 1))
                )
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, season_list):
        # bottom-up mixing (high -> low resolution)
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            # 현재 해상도에서 다음 해상도로 변환
            out_low_res = self.down_sampling_layers[i](out_high)
            # 낮은 해상도와 결합
            out_low = out_low + out_low_res
            out_high = out_low
            
            # 다음 스케일이 있으면 준비
            if i + 2 < len(season_list):
                out_low = season_list[i + 2]
            
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list

class TrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """
    def __init__(self, seq_len: int, d_model: int, down_sampling_layers: int = 2, down_sampling_window: int = 2):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    seq_len // (down_sampling_window ** (i + 1)),
                    seq_len // (down_sampling_window ** i)
                ),
                nn.GELU(),
                nn.Linear(
                    seq_len // (down_sampling_window ** i),
                    seq_len // (down_sampling_window ** i)
                )
            )
            for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list):
        # top-down mixing (low -> high resolution)
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            # 현재 해상도에서 다음 해상도로 변환
            out_high_res = self.up_sampling_layers[i](out_low)
            # 높은 해상도와 결합
            out_high = out_high + out_high_res
            out_low = out_high
            
            # 다음 스케일이 있으면 준비
            if i + 2 < len(trend_list_reverse):
                out_high = trend_list_reverse[i + 2]
            
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list

class TimeMixing(nn.Module):
    """
    Combine season and trend mixing
    """
    def __init__(self, seq_len: int, d_model: int, down_sampling_layers: int = 2, down_sampling_window: int = 2):
        super().__init__()
        self.season_mixing = SeasonMixing(seq_len, d_model, down_sampling_layers, down_sampling_window)
        self.trend_mixing = TrendMixing(seq_len, d_model, down_sampling_layers, down_sampling_window)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.cross_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x_list, decompose_fn):
        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        
        for x in x_list:
            season, trend = decompose_fn(x)
            season = self.cross_layer(season)
            trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # Mixing
        out_season_list = self.season_mixing(season_list)
        out_trend_list = self.trend_mixing(trend_list)

        # Combine results
        output_list = []
        for out_season, out_trend in zip(out_season_list, out_trend_list):
            out = out_season + out_trend
            output_list.append(self.layer_norm(out))

        return output_list