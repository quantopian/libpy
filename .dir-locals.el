((nil . ((eval . (add-to-list 'auto-mode-alist '("\\.h\\'" . c++-mode)))))
 (python-mode . ((fill-column . 79)))
 (c++-mode . ((c-basic-offset . 4)
              (fill-column . 90)
              (flycheck-gcc-language-standard . "gnu++17")
              (eval . (progn
                        (c-set-offset 'innamespace 0)

                        (defun do-shell (s)
                          ;; Helper for running a shell command and getting the first line
                          ;; of its output.
                          (substring (shell-command-to-string s) 0 -1))

                        (setq flycheck-gcc-include-path
                              (let* ((python-include
                                      (do-shell "python -c \"import sysconfig; print(sysconfig.get_path('include'))\""))
                                     (numpy-include
                                      (do-shell "python -c \"import numpy; print(numpy.get_include())\""))
                                     (project-root
                                      (do-shell "git rev-parse --show-toplevel"))
                                     (libpy-include
                                      (concat project-root "/submodules/libpy/include"))
                                     (relative-project-subdirs
                                      (split-string
                                       (shell-command-to-string
                                        "ls -l `git rev-parse --show-toplevel` | grep '^d' | awk '{print $9}'")))
                                     (absolute-project-subdirs
                                      (mapcar
                                       (lambda (subdir) (concat project-root "/" subdir))
                                       relative-project-subdirs)))
                                (append
                                 (list python-include numpy-include libpy-include)
                                 absolute-project-subdirs))))))))
