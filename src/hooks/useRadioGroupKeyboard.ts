"use client";

import { useCallback, useRef } from "react";

export function useRadioGroupKeyboard<T extends string>(
  options: readonly T[],
  value: T | "",
  onChange: (value: T) => void,
) {
  const buttonRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const setButtonRef = useCallback(
    (index: number) => (element: HTMLButtonElement | null) => {
      buttonRefs.current[index] = element;
    },
    [],
  );

  const getTabIndex = useCallback(
    (optionValue: T) => {
      if (!value) {
        return optionValue === options[0] ? 0 : -1;
      }
      return value === optionValue ? 0 : -1;
    },
    [options, value],
  );

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (options.length === 0) return;

      const currentIndex = value ? options.indexOf(value as T) : 0;
      const startIndex = currentIndex >= 0 ? currentIndex : 0;
      let nextIndex = startIndex;

      switch (event.key) {
        case "ArrowRight":
        case "ArrowDown":
          event.preventDefault();
          nextIndex = (startIndex + 1) % options.length;
          break;
        case "ArrowLeft":
        case "ArrowUp":
          event.preventDefault();
          nextIndex = (startIndex - 1 + options.length) % options.length;
          break;
        case "Home":
          event.preventDefault();
          nextIndex = 0;
          break;
        case "End":
          event.preventDefault();
          nextIndex = options.length - 1;
          break;
        default:
          return;
      }

      const nextValue = options[nextIndex];
      onChange(nextValue);
      buttonRefs.current[nextIndex]?.focus();
    },
    [onChange, options, value],
  );

  return { onKeyDown, getTabIndex, setButtonRef };
}
