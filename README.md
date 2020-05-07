# Amortizing action

_The code for the SAC algorithm was built on top of this [curl](https://github.com/MishaLaskin/curl/blob/master/curl_sac.py) repo_
_This code can be adapated for pixels, but is currently focused on states_

**Check standard deviation, it had been changed**
**Uses weight scheme **

## Install
- `conda` environment (`curl`) taken from the `conda_env.yml` file 
- `torch` has been downgraded `conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch`, and `tensorboard` has been removed

## Running
```bash
git clone https://github.com/alec-tschantz/mbmf.git
cd mbmf
conda activate curl
python scripts/sac_script.py
```
## Scripts

- `sac_script.py` - train a SAC agent
- `mpc_script.py` - train an MPC agent
- `hybrid_script.py` - train a hybrid agent
- `test_script.py` - test hybrid agent with trained SAC & ensemble model
