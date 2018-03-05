import json
import random
import numpy as np
import pandas as pd

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(1)

'''
Data format:
    - radiant_win
    - start_time
    - duration
    - avg_mmr
    - num_mmr
    - lobby_type
    - radiant_heroes
    - dire_heroes

Info:
Lobby types:
    - https://github.com/odota/dotaconstants/blob/master/json/lobby_type.json
Game modes:
    - https://github.com/odota/dotaconstants/blob/master/json/game_mode.json
Patch dates:
    - Feb 1, 2018: 7.08
    - Feb 15, 2018: 7.09
    - Mar 1, 2018: 7.10
'''

#### Functions #################################################################

def parse_team(team, randomize, radiant=True):
    colnames = [
        'radiant_' + str(idx) if radiant else 'dire_' + str(idx)
        for idx, _ in enumerate(team)
    ]
    picks = [int(x) for x in team.split(',')]
    if radiant:
        random.shuffle(picks)
    return dict(zip(colnames, picks))


def get_patch(timestamp):
    # Mar 1: 1519891200
    # Feb 15: 1518681600
    # Feb 1: 1517472000
    if timestamp > 1519891200:
        return '7.10'
    if timestamp > 1518681600:
        return '7.09'
    if timestamp > 1517472000:
        return '7.08'
    return None


def load_data(path):
    record_list = []
    with open(path, 'r') as f:
        for line in f:
            try:
                record = {}
                j = json.loads(line)
                radiant = parse_team(
                    j['radiant_team'],
                    randomize=True,
                )
                dire = parse_team(
                    j['dire_team'],
                    randomize=True,
                    radiant=False,
                )
                record.update(radiant)
                record.update(dire)
                record['radiant_win'] = j['radiant_win']
                record['start_time'] = j['start_time']
                record['duration'] = j['duration']
                record['avg_mmr'] = j['avg_mmr']
                record['num_mmr'] = j['num_mmr']
                record['lobby_type'] = j['lobby_type']
                record['game_mode'] = j['game_mode']
                record['patch'] = get_patch(j['start_time'])
                record_list.append(record)
            except:
                print(line)
    return pd.DataFrame(record_list)


#### Load and process data #####################################################


df = load_data('data/reg_matches.json')

df = df.loc[df['avg_mmr'] < 2000]

X_pd = df[[
    'dire_0',
    'dire_1',
    'dire_2',
    'dire_3',
    'dire_4',
    'radiant_0',
    'radiant_1',
    'radiant_2',
    'radiant_3',
    'radiant_4',
]].dropna()

y = X_pd['radiant_0']
X = X_pd.drop(['radiant_0'], axis=1)

dl = DataLoader(
    TensorDataset(
        torch.LongTensor(np.array(X)),
        torch.LongTensor(np.array(y)),
    ),
    batch_size=1,
    shuffle=True,
)


#### Model #####################################################################

class DotaEmbedding(nn.Module):
    def __init__(
        self,
        n_heroes,
        embedding_dim,
        context_size=9,
        hidden_size=64,
    ):
        super(DotaEmbedding, self).__init__()
        self.embeddings = nn.Embedding(n_heroes, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_heroes)

    def forward(self, context):
        embeds = self.embeddings(context).view((1, -1))
        out = F.relu(self.fc1(embeds))
        out = self.fc2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = DotaEmbedding(121, 2, hidden_size=128)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(1):
    total_loss = 0
    for idx, batch in enumerate(dl):
        context = autograd.Variable(batch[0])
        target = autograd.Variable(batch[1])
        model.zero_grad()

        log_probs = model(context)

        loss = loss_function(
            log_probs,
            target,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.data.numpy()[0]
        if idx % 10000 == 0:
            print(idx)
    print(
        'Epoch {epoch}, loss {loss}'.format(
            epoch=epoch,
            loss=total_loss / X.shape[0],
        )
    )
    losses.append(total_loss)
print(losses)


#### Process for viz ###########################################################

heroes = load_heroes()
hero_data_dict = {x['id']: x for x in heroes}
hero_dict = {x['id']: x['localized_name'] for x in heroes}

all_roles = []
for val in hero_data_dict.values():
    all_roles.extend(val['roles'])
all_roles = list(set(all_roles))

trained_embeddings = model.embeddings

data = []

for idx, name in hero_dict.items():
    embed = trained_embeddings(autograd.Variable(torch.LongTensor([idx])))
    embed = embed.data.numpy()[0]
    datum = [name, embed[0], embed[1]]
    hero_roles = set(hero_data_dict[idx]['roles'])
    for role in all_roles:
        if role in hero_roles:
            datum.append(1)
        else:
            datum.append(0)
    data.append(datum)

out_df = pd.DataFrame(data)
out_df.columns = ['name', 'x', 'y'] + all_roles
out_df.to_csv('data/viz.csv')
