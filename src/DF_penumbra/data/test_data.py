# false negative from test2 regardless of the model 
emails = ['Manuel826@yahoo.com', 'muckpatrick@gmail.com']

# false positive from rf-predict-all
f_pos = [
    ('po_co_342', 'po_co_928'), ('po_no_171', 'po_no_486'), ('po_co_319', 'po_no_490'),
    ('po_no_400', 'po_no_401'), ('po_no_481', 'po_no_485'), ('po_no_677', 'po_no_681'),
    ('po_no_331', 'po_no_413'), ('po_no_413', 'po_no_197'), ('po_no_114', 'po_no_133'),
    ('po_no_114', 'po_no_129')
]

# false negative from rf-predict-all

# the cases with the matching last names but different first names (sb's manual check) 
fname_cases = [
    ('sp_al_389', 'sp_va_136', 'po_us_355', 'po_vc_341'), 
    ('po_co_808', 'po_us_301', 'po_vc_288'),
    ('po_vc_167', 'po_co_75', 'po_us_175'),
    ('sp_al_13', 'sp_al_319', 'sp_al_368', 'sp_al_533', 'sp_ne_7', 'po_co_624', 'po_us_262', 'po_vc_249'),
    ('po_co_96', 'po_no_406'),
    ('po_vc_241', 'sp_al_117'),
    ('po_co_931', 'po_us_318', 'po_vc_304'),
    ('sp_al_52', 'sp_al_307', 'sp_va_89', 'po_co_729', 'po_us_283', 'po_vc_270'),
    ('po_co_655', 'po_us_144', 'po_vc_140'),
    ('sp_al_503', 'sp_ne_88', 'po_us_366', 'po_vc_352'),
    ('sp_al_43', 'sp_va_146', 'po_co_676', 'po_us_42', 'po_vc_41'),
    ('po_vc_194', 'sp_al_181', 'sp_al_259', 'sp_va_93', 'po_co_324', 'po_us_205'),
    ('po_co_843', 'po_us_133', 'po_vc_130'),
    ('sp_al_302', 'sp_al_343', 'sp_al_376', 'sp_al_387', 'sp_al_419', 'sp_al_456', 'sp_al_501', 'sp_al_526', 'sp_va_14', 'po_co_432', 'po_us_110', 'po_vc_108')
]

# synonym name cases (sb's manual check)
syn_cases = [
    ('po_vc_142', 'po_co_975', 'po_us_146'),
    ('sp_al_120', 'sp_ne_28', 'po_co_131', 'po_us_60', 'po_vc_59'),
    ('po_us_248', 'po_vc_235'),
    ('po_us_165', 'po_vc_159'),
    ('po_no_660', 'po_no_662'),
    ('sp_al_352', 'sp_al_541', 'sp_va_100', 'po_co_586', 'po_vc_152', 'po_us_158', 'po_no_378'),
    ('po_ou_211', 'po_co_739', 'po_co_743'),
    ('po_co_330', 'po_us_207', 'po_vc_196')
]

# weak matching cases (sb's manual check)
weak_cases = [
    ('sp_al_365', 'sp_al_525', 'sp_al_575', 'sp_va_53', 'po_vc_362', 'po_us_376'),
    ('po_co_886', 'po_no_746'),
    ('sp_al_329', 'sp_va_128'),
    ('sp_al_613', 'sp_ne_42'),
    ('po_co_721', 'sp_al_246', 'sp_ne_34'),
    ('po_vc_340', 'sp_al_363'),
    ('po_no_149', 'sp_al_221', 'sp_ne_82', 'po_co_364'),
    ('po_ou_30', 'po_co_1030'),
    ('sp_al_199', 'sp_ne_38', 'po_co_451'),
    ('sp_al_222', 'sp_ne_50', 'po_co_862'),
    ('po_ou_27', 'po_co_976'),
    ('sp_al_139', 'sp_ne_79', 'po_ou_20', 'po_co_263'),
    ('sp_al_524', 'sp_va_131', 'po_vc_326', 'po_us_340'),
    ('sp_al_564', 'sp_va_170', 'po_no_4'),
    ('sp_al_658', 'sp_al_660', 'sp_ne_22', 'po_ou_24', 'po_co_882'),
    ('sp_al_480', 'sp_al_590', 'sp_al_597', 'sp_ne_9', 'po_no_198'),
    ('sp_al_270', 'sp_ne_49', 'po_co_731'),
    ('sp_al_420', 'sp_va_153', 'po_vc_325', 'po_us_339'),
    ('sp_al_410', 'sp_va_192', 'po_vc_357', 'po_us_371'),
    ('sp_al_287', 'sp_ne_60', 'po_no_845'),
    ('sp_al_279', 'sp_ne_59', 'po_co_378'),
    ('sp_al_251', 'sp_ne_74', 'po_co_759'),
    ('sp_al_203', 'sp_al_664', 'sp_ne_20', 'po_co_355', 'po_co_393'),
    ('sp_al_628', 'sp_ne_48', 'po_co_701', 'po_vc_267', 'po_us_280'),
    ('po_ou_29', 'po_co_872'),
    ('sp_al_451', 'sp_al_452', 'sp_va_90', 'po_co_632')
]