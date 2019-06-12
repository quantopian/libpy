import sysconfig
print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')
