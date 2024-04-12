from decouple import AutoConfig
import ast
import sys
import os 

class AutoConfigImpl(AutoConfig):
    def __init__(self, search_path=None):
        super(AutoConfigImpl, self).__init__(search_path)
        self.search_path = search_path
        self._load(self.search_path or self._caller_path())
        # verify "type ="
        for option, value in self.config.repository.data.items():
            if 'type=' in value:
                values = value.split(',type=')
                value = values[0]
                value = value.strip()
                type = values[1]
                if type == 'int':
                    value = int(value)
                elif type == 'list':
                    value = ast.literal_eval(value)
                elif type =='bool':
                    value = True if value =='True' else False
                self.config.repository.data[option] = value

DIR_DUPLICATEFINDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"DIR_DUPLICATEFINDER: {DIR_DUPLICATEFINDER}")
sys.path.append(DIR_DUPLICATEFINDER)

env_file = DIR_DUPLICATEFINDER + '/.env'
config = AutoConfigImpl(env_file)
assert os.path.exists(env_file), f"We dont have env at {env_file}"
this_module = sys.modules[__name__]
for option, value in config.config.repository.data.items():
    setattr(this_module, option, value)

