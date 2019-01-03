((c-mode . ((mode . c++)
            (eval . (setq flycheck-gcc-include-path
                          (list
                           "../"
                           "../include"
                           "../../include"
                           "../../../include"
                           "../submodules/googletest/googletest/include"
                           (expand-file-name "~/.virtualenvs/factset/include/python3.6m")
                           (expand-file-name "~/.virtualenvs/factset/lib/python3.6/site-packages/numpy/core/include")
                           "/usr/include/python3.6m")))
            (eval . (set-fill-column 90))
            (eval . (c-set-offset 'innamespace 0))))
 (c++-mode . ((eval . (setq flycheck-gcc-include-path
                            (list
                             "../"
                             "../include"
                             "../../include"
                             "../../../include"
                             "../submodules/googletest/googletest/include"
                             (expand-file-name "~/.virtualenvs/factset/include/python3.6m")
                             (expand-file-name "~/.virtualenvs/factset/lib/python3.6/site-packages/numpy/core/include")
                             "/usr/include/python3.6m")))
              (eval . (c-set-offset 'innamespace 0))
              (eval . (set-fill-column 90))
              (flycheck-gcc-language-standard . "gnu++17"))))
