import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Player Analysis",
  description: "Sports video analysis platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
