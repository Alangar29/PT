# -*- coding: utf-8 -*-
"""Pruebas de la logica de la ventana de contexto (no requiere camara ni APIs).

    python test_ventana_contexto.py
"""
import unittest

from context_window import (
    extract_context_window, ends_sentence, is_trailing_punct_only,
    window_text_marked,
)


def make_words(text, para_id=0, start=0):
    return [{"idx": start + k, "text": t, "para_id": para_id}
            for k, t in enumerate(text.split())]


def index_of(words, text):
    return next(k for k, w in enumerate(words) if w["text"] == text)


class TestVentana(unittest.TestCase):

    def test_ejemplo_break_up(self):
        words = make_words(
            "After ten years, they decided to break up and sell the house.")
        i = index_of(words, "up")
        w = extract_context_window(words, i)
        self.assertEqual([x["text"] for x in w["before"]],
                         ["decided", "to", "break"])
        self.assertEqual([x["text"] for x in w["after"]],
                         ["and", "sell", "the"])
        self.assertEqual(w["target_pos"], 3)
        self.assertEqual(w["text"], "decided to break up and sell the")
        self.assertEqual(window_text_marked(w),
                         "decided to break [up] and sell the")

    def test_paro_hacia_atras_por_fin_de_oracion(self):
        # "late." cierra la oracion anterior: hacia atras solo llega "She"
        words = make_words("It was late. She gave up soon and left")
        i = index_of(words, "up")
        w = extract_context_window(words, i)
        self.assertEqual([x["text"] for x in w["before"]], ["She", "gave"])
        self.assertEqual([x["text"] for x in w["after"]],
                         ["soon", "and", "left"])

    def test_paro_hacia_adelante_por_fin_de_oracion(self):
        # "up." termina la oracion: no se toma nada hacia adelante
        words = make_words("I will break up. Then we leave")
        i = index_of(words, "break")
        w = extract_context_window(words, i)
        self.assertEqual([x["text"] for x in w["after"]], ["up."])
        i = index_of(words, "up.")
        w = extract_context_window(words, i)
        self.assertEqual(w["after"], [])
        self.assertEqual(w["text"], "I will break up.")

    def test_menos_de_tres_a_un_lado(self):
        words = make_words("Go up now")
        w = extract_context_window(words, 1)
        self.assertEqual(w["text"], "Go up now")
        self.assertEqual(w["target_pos"], 1)

    def test_palabra_sola(self):
        words = make_words("Hello")
        w = extract_context_window(words, 0)
        self.assertEqual(w["text"], "Hello")
        self.assertEqual(w["before"], [])
        self.assertEqual(w["after"], [])

    def test_inicio_y_fin_de_lista(self):
        words = make_words("one two three four five")
        w = extract_context_window(words, 0)
        self.assertEqual(w["text"], "one two three four")
        w = extract_context_window(words, 4)
        self.assertEqual(w["text"], "two three four five")

    def test_cambio_de_parrafo(self):
        a = make_words("this is the end of a paragraph", para_id=0)
        b = make_words("and this starts another one", para_id=1, start=len(a))
        words = a + b
        # senalada = "paragraph" (ultima del parrafo 0): no toma del parrafo 1
        i = index_of(words, "paragraph")
        w = extract_context_window(words, i)
        self.assertEqual(w["after"], [])
        # senalada = "and" (primera del parrafo 1): no toma del parrafo 0
        i = index_of(words, "and")
        w = extract_context_window(words, i)
        self.assertEqual(w["before"], [])
        self.assertEqual(w["text"], "and this starts another")

    def test_la_coma_no_detiene(self):
        words = make_words("After ten years, they decided to leave")
        i = index_of(words, "they")
        w = extract_context_window(words, i)
        # la coma de "years," no corta el recorrido hacia atras
        self.assertEqual([x["text"] for x in w["before"]],
                         ["After", "ten", "years,"])
        self.assertEqual([x["text"] for x in w["after"]],
                         ["decided", "to", "leave"])

    def test_comillas_de_cierre(self):
        self.assertTrue(ends_sentence("house."))
        self.assertTrue(ends_sentence("stop!\""))
        self.assertTrue(ends_sentence("really?)"))
        self.assertFalse(ends_sentence("years,"))
        self.assertFalse(ends_sentence("up"))
        self.assertFalse(ends_sentence('"'))

    def test_puntuacion_suelta(self):
        self.assertTrue(is_trailing_punct_only("."))
        self.assertTrue(is_trailing_punct_only(".\""))
        self.assertFalse(is_trailing_punct_only("a."))
        self.assertFalse(is_trailing_punct_only("("))


if __name__ == "__main__":
    unittest.main(verbosity=2)
