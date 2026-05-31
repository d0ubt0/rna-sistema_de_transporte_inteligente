import torch
import torch.nn as nn


class TransportLSTM(nn.Module):
    """
    LSTM con atención para predicción de demanda.

    Features temporales:
        pasajeros
        dia_semana
        mes
        festivo

    Variables categóricas:
        ruta
        clima
    """

    def __init__(
        self,
        num_routes,
        num_climas,
        input_size=4,
        hidden_size=160,
        num_layers=2,
        route_embedding_dim=12,
        clima_embedding_dim=6,
        dropout=0.25,
        bidirectional=True,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        lstm_width = hidden_size * (2 if bidirectional else 1)

        self.input_norm = nn.LayerNorm(input_size)

        self.route_embedding = nn.Embedding(
            num_routes,
            route_embedding_dim,
        )

        self.clima_embedding = nn.Embedding(
            num_climas,
            clima_embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_width, lstm_width // 2),
            nn.Tanh(),
            nn.Linear(lstm_width // 2, 1),
        )

        fc_input_size = (
            lstm_width
            + route_embedding_dim
            + clima_embedding_dim
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(fc_input_size),
            nn.Linear(fc_input_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x,
        route_ids,
        clima_ids,
    ):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)

        attention_scores = self.temporal_attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        route_embed = self.route_embedding(route_ids)

        clima_embed = self.clima_embedding(clima_ids)

        combined = torch.cat(
            [
                context,
                route_embed,
                clima_embed,
            ],
            dim=1,
        )

        return self.fc(combined)
