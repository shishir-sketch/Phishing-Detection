from phishing_detector.features import FEATURE_NAMES, extract_features_df, extract_url_features


def test_ip_host_detected():
    f = extract_url_features("http://192.168.10.5/login.php")
    assert f["has_ip_host"] == 1
    assert f["uses_https"] == 0


def test_https_and_clean_domain():
    f = extract_url_features("https://www.example.com/about")
    assert f["uses_https"] == 1
    assert f["has_ip_host"] == 0
    assert f["num_subdomains"] == 1
    assert f["suspicious_keyword_count"] == 0


def test_at_symbol_and_brand_misuse():
    f = extract_url_features("http://paypal.secure-login.evil.xyz/verify/account")
    assert f["brand_in_subdomain_or_path"] == 1
    assert f["suspicious_tld"] == 1
    assert f["suspicious_keyword_count"] >= 3
    assert extract_url_features("http://paypal.com@evil.xyz/")["count_at"] == 1


def test_brand_on_its_own_domain_is_not_flagged():
    assert extract_url_features("https://www.paypal.com/signin")["brand_in_subdomain_or_path"] == 0


def test_two_level_tld_subdomain_count():
    assert extract_url_features("https://www.bbc.co.uk/news")["num_subdomains"] == 1


def test_missing_scheme_and_garbage_do_not_crash():
    assert extract_url_features("example.com/path")["hostname_length"] == len("example.com")
    for bad in ["", "http://[::1", "   ", "ht!tp://%%%"]:
        assert set(extract_url_features(bad)) == set(FEATURE_NAMES)


def test_dataframe_shape_and_order():
    df = extract_features_df(["http://a.com", "https://b.org/x?y=1"])
    assert list(df.columns) == FEATURE_NAMES
    assert df.shape == (2, len(FEATURE_NAMES))
    assert not df.isna().any().any()
