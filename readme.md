## Setup

### First Time

#### Installing, Starting the Virtual Environment

```bash
python3 -m venv venv    # creates folder “venv” for virtual env
. venv/bin/activate     # hop into env
pip install -r requirements.txt
```

<!-- #### Load up your environment variables

- `cp .envrc{.sample,}`
- Got to `.envrc` and fill in missing keys
- Install [direnv](https://direnv.net/) and that will take care of it for you!
  - Optionally, you can run `source .envrc` for every terminal, but this approach _is not recommended_. Every person who has used this has inevitably forgot to do it and has run into issues -->
