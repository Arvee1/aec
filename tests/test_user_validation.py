import unittest

from user_validation import validate_name


class ValidateNameTests(unittest.TestCase):
    def test_rejects_blank_name(self):
        with self.assertRaises(ValueError):
            validate_name(" \t  ")

    def test_rejects_name_shorter_than_two_characters(self):
        with self.assertRaises(ValueError):
            validate_name("A")

    def test_rejects_name_longer_than_100_characters(self):
        with self.assertRaises(ValueError):
            validate_name("A" * 101)

    def test_rejects_digits_and_symbols(self):
        with self.assertRaises(ValueError):
            validate_name("Ryan123!")

    def test_rejects_names_without_any_letters(self):
        for name in ("--", "ाा"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_name(name)

    def test_normalizes_outer_and_repeated_whitespace(self):
        self.assertEqual(validate_name("  Jean\t  Luc  "), "Jean Luc")

    def test_accepts_international_letters_apostrophes_and_hyphens(self):
        self.assertEqual(validate_name("Zoë O’Neil-Smith"), "Zoë O’Neil-Smith")

    def test_accepts_non_latin_names_with_combining_marks(self):
        self.assertEqual(validate_name("राम"), "राम")

    def test_normalizes_decomposed_unicode_letters(self):
        self.assertEqual(validate_name("Jose\u0301"), "José")


if __name__ == "__main__":
    unittest.main()
